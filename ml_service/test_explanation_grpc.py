"""Exercise the Go-facing RPC through a real local gRPC server and mock HTTPS."""

import json
import logging
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

import grpc
from google.rpc import error_details_pb2
from grpc_status import rpc_status

from .logic import YandexAIStudioProvider
from .models import ModelManager
from .server import JSONFormatter, create_server
from .test_yandex import ENV, response
from .v1 import service_pb2 as pb
from .v1 import service_pb2_grpc as pb_grpc


def request():
    return pb.ExplainCalculationRequest(
        request_id="req-explanation", calculation_id="calc1",
        calculation_kind=pb.CALCULATION_KIND_DV01, currency="RUB",
        total_rub="800", data_as_of="2026-11-17",
        positions=[pb.ExplanationPosition(position_id="bond1", side="buy", value_rub="800")],
    )


class ExplanationRPCTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.models = ModelManager(Path(self.temporary.name))
        self.provider = YandexAIStudioProvider(
            "https://ai.api.cloud.yandex.net/v1/chat/completions",
            "gpt://test-folder/deepseek-v4.1-flash", ENV["YANDEX_API_KEY"],
            timeout=1, retries=0, project="test-folder", auth_scheme="Api-Key", structured_output=True,
        )

    def stub(self, provider):
        server, port = create_server(self.models, provider, "127.0.0.1:0")
        server.start()
        self.addCleanup(lambda: server.stop(0).wait())
        channel = grpc.insecure_channel(f"127.0.0.1:{port}")
        self.addCleanup(channel.close)
        return pb_grpc.MLServiceStub(channel)

    def assert_rpc_code(self, code, call):
        with self.assertRaises(grpc.RpcError) as error:
            call()
        self.assertEqual(error.exception.code(), code)
        self.assertNotIn(ENV["YANDEX_API_KEY"], error.exception.details())
        rich = rpc_status.from_call(error.exception)
        detail = error_details_pb2.ErrorInfo()
        rich.details[0].Unpack(detail)
        self.assertEqual(detail.metadata["method"], "ExplainCalculation")
        self.assertEqual(detail.metadata["request_id"], "req-explanation")

    def test_yandex_roundtrip_serialization_minimal_facts_and_logs(self):
        stub = self.stub(self.provider)
        with patch("urllib.request.OpenerDirector.open", return_value=response()) as send, self.assertLogs(
            "ml_service.server", level="INFO",
        ) as logs:
            result = stub.ExplainCalculation(request(), timeout=3)
        self.assertTrue(result.verified)
        self.assertEqual(result.calculation_id, "calc1")
        self.assertEqual(result.request_id, "req-explanation")
        self.assertEqual(result.fact_ids, ["total_rub"])
        payload = json.loads(send.call_args.args[0].data)
        facts = json.loads(payload["messages"][1]["content"])
        self.assertNotIn("request_id", facts)
        self.assertEqual(facts["positions"][0]["side"], "BUY")
        self.assertEqual(facts["facts"]["total_rub"], "800")
        formatter = JSONFormatter()
        rendered = "\n".join(formatter.format(record) for record in logs.records)
        self.assertNotIn(ENV["YANDEX_API_KEY"], rendered)
        self.assertNotIn("800", rendered)
        self.assertEqual(json.loads(rendered)["rpc_method"], "ExplainCalculation")

    def test_unknown_facts_and_wrong_numbers_are_not_returned(self):
        stub = self.stub(self.provider)
        for output in ({"text": "Итог 801 RUB", "fact_ids": ["total_rub"]},
                       {"text": "Итог 800 RUB", "fact_ids": ["unknown"]}):
            with self.subTest(output=output), patch("urllib.request.OpenerDirector.open", return_value=response(output)):
                self.assert_rpc_code(grpc.StatusCode.DATA_LOSS, lambda: stub.ExplainCalculation(request(), timeout=3))

    def test_missing_provider_timeout_and_network_error(self):
        absent = self.stub(None)
        self.assert_rpc_code(grpc.StatusCode.FAILED_PRECONDITION, lambda: absent.ExplainCalculation(request(), timeout=3))
        stub = self.stub(self.provider)
        for error in (TimeoutError(), urllib.error.URLError("offline")):
            with self.subTest(error=error), patch("urllib.request.OpenerDirector.open", side_effect=error):
                self.assert_rpc_code(grpc.StatusCode.UNAVAILABLE, lambda: stub.ExplainCalculation(request(), timeout=3))

    def test_invalid_facts_are_rejected_before_paid_call(self):
        stub = self.stub(self.provider)
        bad = request()
        bad.total_rub = "801"
        with patch("urllib.request.OpenerDirector.open") as send:
            self.assert_rpc_code(grpc.StatusCode.INVALID_ARGUMENT, lambda: stub.ExplainCalculation(bad, timeout=3))
            send.assert_not_called()

    def test_readiness_and_observed_metric_do_not_trigger_llm(self):
        stub = self.stub(self.provider)
        with patch("urllib.request.OpenerDirector.open") as send:
            ready = stub.GetReadiness(pb.ReadinessRequest(), timeout=3)
            self.assertTrue(ready.explanation_ready)
            self.assertFalse(ready.metric_ready)
            value = stub.EstimateMissingMetric(pb.EstimateMissingMetricRequest(
                request_id="req-observed", instrument_id="bond1", as_of_date="2026-11-17",
                metric=pb.METRIC_KIND_DURATION_MOD, observed_value=2,
            ), timeout=3)
            self.assertEqual(value.value, 2)
            self.assertEqual(value.source, pb.METRIC_SOURCE_OBSERVED)
            send.assert_not_called()

    def test_unavailable_service_has_controlled_grpc_status(self):
        server, port = create_server(self.models, None, "127.0.0.1:0")
        server.start()
        server.stop(0).wait()
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel, self.assertRaises(grpc.RpcError) as error:
            pb_grpc.MLServiceStub(channel).GetReadiness(pb.ReadinessRequest(), timeout=1)
        self.assertEqual(error.exception.code(), grpc.StatusCode.UNAVAILABLE)


if __name__ == "__main__":
    logging.basicConfig(level=logging.CRITICAL)
    unittest.main()
