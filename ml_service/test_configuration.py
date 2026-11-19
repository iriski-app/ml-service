"""Saved AI Studio configurations keep model and instruction outside Python."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import grpc

from .logic import ServiceError, YandexConfigurationProvider, provider_from_env
from .models import ModelManager
from .server import create_server
from .test_explanation_grpc import request
from .test_logic import DATA
from .test_yandex import ENV
from .v1 import service_pb2_grpc as rpc


def configuration_response(**overrides):
    body = {
        "status": "completed", "model": "gpt://test-folder/deepseek-v4.1-flash",
        "error": None, "incomplete_details": None,
        "output": [{
            "type": "message", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": json.dumps({
                "text": "DV01: 800 RUB", "fact_ids": ["total_rub"],
            })}],
        }],
    }
    body.update(overrides)
    response = MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps(body).encode()
    return response


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {
            **ENV, "LLM_PROVIDER": "yandex_configuration", "YANDEX_PROMPT_ID": "saved-config-1",
        }, clear=True)
        env.start()
        self.addCleanup(env.stop)

    def test_payload_uses_configuration_without_local_instruction_or_model(self):
        provider = provider_from_env()
        self.assertIsInstance(provider, YandexConfigurationProvider)
        with patch("urllib.request.OpenerDirector.open", return_value=configuration_response()) as send:
            answer = provider.explain(DATA)
        request = send.call_args.args[0]
        self.assertEqual(request.full_url, "https://ai.api.cloud.yandex.net/v1/responses")
        self.assertEqual(request.get_header("Authorization"), "Api-Key " + ENV["YANDEX_API_KEY"])
        self.assertEqual(request.get_header("Openai-project"), ENV["YANDEX_FOLDER_ID"])
        body = json.loads(request.data)
        self.assertEqual(body["prompt"], {"id": "saved-config-1"})
        self.assertEqual(json.loads(body["input"]), DATA)
        for override in ("instructions", "messages", "model", "reasoning", "reasoning_effort", "max_output_tokens"):
            self.assertNotIn(override, body)
        self.assertFalse(body["store"])
        self.assertFalse(body["stream"])
        self.assertNotIn("previous_response_id", body)
        self.assertNotIn("conversation", body)
        self.assertEqual(body["tool_choice"], "none")
        self.assertEqual(body["tools"], [])
        self.assertEqual(body["text"]["format"]["type"], "json_schema")
        self.assertEqual(answer["text"], "DV01: 800 RUB")

    def test_missing_config_never_falls_back_to_direct_model(self):
        with patch.dict(os.environ, {"YANDEX_PROMPT_ID": ""}), patch("urllib.request.OpenerDirector.open") as send:
            with self.assertRaises(ServiceError) as error:
                provider_from_env()
            self.assertEqual(error.exception.code, "FAILED_PRECONDITION")
            send.assert_not_called()

    def test_version_is_optional_and_no_model_setting_is_required(self):
        with patch.dict(os.environ, {"YANDEX_PROMPT_VERSION": "2", "YANDEX_MODEL": ""}), patch(
            "urllib.request.OpenerDirector.open", return_value=configuration_response(),
        ) as send:
            provider_from_env().explain(DATA)
        self.assertEqual(json.loads(send.call_args.args[0].data)["prompt"], {"id": "saved-config-1", "version": "2"})

    def test_invalid_configuration_settings(self):
        for setting in ({"YANDEX_PROMPT_ID": "bad/id"}, {"YANDEX_PROMPT_VERSION": "bad\nversion"},
                        {"LLM_TIMEOUT_SECONDS": "invalid"}, {"LLM_RETRIES": "invalid"}):
            with self.subTest(setting=setting), patch.dict(os.environ, setting):
                with self.assertRaises(ServiceError) as error:
                    provider_from_env()
                self.assertEqual(error.exception.code, "FAILED_PRECONDITION")

    def test_incomplete_refused_and_tool_responses_are_rejected(self):
        for overrides in (
            {"status": "incomplete"}, {"status": "failed"}, {"error": {"code": "error"}},
            {"incomplete_details": {"reason": "max_output_tokens"}}, {"output": []},
            {"output": [{"type": "function_call"}]},
            {"output": [{"type": "message", "role": "assistant", "status": "completed",
                         "content": [{"type": "refusal", "refusal": "no"}]}]},
        ):
            with self.subTest(overrides=overrides), patch(
                "urllib.request.OpenerDirector.open", return_value=configuration_response(**overrides),
            ):
                with self.assertRaises(ServiceError) as error:
                    provider_from_env().explain(DATA)
                self.assertEqual(error.exception.code, "DATA_LOSS")

    def test_grpc_roundtrip_and_validation_for_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            server, port = create_server(ModelManager(Path(directory)), provider_from_env(), "127.0.0.1:0")
            server.start()
            try:
                with grpc.insecure_channel(f"127.0.0.1:{port}") as channel, patch(
                    "urllib.request.OpenerDirector.open", return_value=configuration_response(),
                ):
                    answer = rpc.MLServiceStub(channel).ExplainCalculation(request(), timeout=3)
                    self.assertTrue(answer.verified)
                    self.assertEqual(answer.calculation_id, "calc1")
                    self.assertEqual(answer.fact_ids, ["total_rub"])
                wrong = configuration_response(output=[{
                    "type": "message", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": json.dumps({
                        "text": "DV01: 801 RUB", "fact_ids": ["total_rub"],
                    })}],
                }])
                with grpc.insecure_channel(f"127.0.0.1:{port}") as channel, patch(
                    "urllib.request.OpenerDirector.open", return_value=wrong,
                ), self.assertRaises(grpc.RpcError) as error:
                    rpc.MLServiceStub(channel).ExplainCalculation(request(), timeout=3)
                self.assertEqual(error.exception.code(), grpc.StatusCode.DATA_LOSS)
            finally:
                server.stop(0).wait()


if __name__ == "__main__":
    unittest.main()
