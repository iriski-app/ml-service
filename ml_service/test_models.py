"""Contract tests use tiny disposable CatBoost artifacts, never production data."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

import grpc
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from google.rpc import error_details_pb2
from grpc_health.v1 import health_pb2, health_pb2_grpc
from grpc_status import rpc_status

from .models import (
    IMPUTATION_CATEGORICAL,
    IMPUTATION_FEATURES,
    SCENARIO_FEATURES,
    ModelError,
    ModelManager,
)
from .server import create_server
from .v1 import service_pb2 as pb
from .v1 import service_pb2_grpc as pb_grpc


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_artifacts(root: Path) -> None:
    metric_dir = root / "imputation"
    scenario_dir = root / "scenario"
    metric_dir.mkdir()
    scenario_dir.mkdir()
    data = []
    for i in range(10):
        data.append({
            "years_to_maturity": 1 + i / 2,
            "years_to_yield_date": 1 + i / 2,
            "coupon_type_asof": "fixed" if i % 2 else "zero",
            "coupon_freq_asof": 2.0,
            "coupon_period_days_asof": 182.0,
            "coupon_percent": 8.0,
            "coupon_value": 40.0,
            "currency_asof": "RUB",
            "face_value": 1000.0,
            "ytm": 10.0 + i,
            "has_buyback": 0.0,
            "is_government": 1.0,
        })
    frame = pd.DataFrame(data, columns=IMPUTATION_FEATURES)
    filenames = {}
    interval_filenames = {}
    for name, values in {
        "duration_mod": [1 + i / 2 for i in range(10)],
        "convexity": [2 + i for i in range(10)],
    }.items():
        model = CatBoostRegressor(iterations=12, depth=2, learning_rate=0.2, loss_function="RMSE",
                                  verbose=False, allow_writing_files=False, thread_count=1)
        model.fit(frame, np.log1p(values), cat_features=IMPUTATION_CATEGORICAL)
        filenames[name] = f"{name}.cbm"
        model.save_model(str(metric_dir / filenames[name]))
        quantile = CatBoostRegressor(
            iterations=12, depth=2, learning_rate=0.2,
            loss_function="MultiQuantile:alpha=0.1,0.5,0.9",
            verbose=False, allow_writing_files=False, thread_count=1,
        )
        quantile.fit(frame, np.log1p(values), cat_features=IMPUTATION_CATEGORICAL)
        interval_filenames[name] = f"{name}_quantiles.cbm"
        quantile.save_model(str(metric_dir / interval_filenames[name]))
    _write_json(metric_dir / "metadata.json", {
        "features": IMPUTATION_FEATURES,
        "categorical": IMPUTATION_CATEGORICAL,
        "target_transform": "log1p",
        "model_files": filenames,
        "interval_model_files": interval_filenames,
        "calibration_radii_log": {
            f"{name}:{regime}": 0.1
            for name in filenames
            for regime in ("available_features", "without_yield")
        },
    })
    _write_json(metric_dir / "deployment.json", {
        "model_version": "test-1", "supported_currencies": ["RUB"],
        "years_to_yield_date_min": 0.5, "years_to_yield_date_max": 10,
        "sha256": {filename: _hash(metric_dir / filename) for filename in [*filenames.values(), *interval_filenames.values()]},
    })
    scenario_frame = pd.DataFrame([
        {"shock_abs": (i - 5) / 1000, "DurationMod": 1 + i / 10,
         "YIELDCLOSE": 10 + i, "FACEVALUE": 1000.0}
        for i in range(10)
    ], columns=SCENARIO_FEATURES)
    scenario_target = -scenario_frame["DurationMod"] * scenario_frame["shock_abs"] * 10000
    model = CatBoostRegressor(iterations=12, depth=2, learning_rate=0.2, loss_function="RMSE",
                              verbose=False, allow_writing_files=False, thread_count=1)
    model.fit(scenario_frame, scenario_target)
    model.save_model(str(scenario_dir / "scenario.cbm"))
    _write_json(scenario_dir / "metadata.json", {
        "features": SCENARIO_FEATURES, "model_file": "scenario.cbm",
        "sha256": {"scenario.cbm": _hash(scenario_dir / "scenario.cbm")},
        "model_version": "test-1", "target_kind": "formula_proxy_reference_position",
        "target_unit": "RUB_FOR_10_FACE_VALUES", "shift_bp_min": -200,
        "shift_bp_max": 200, "approved_for_service": True, "experimental": True,
    })


def metric_request() -> pb.EstimateMissingMetricRequest:
    return pb.EstimateMissingMetricRequest(
        request_id="req1", instrument_id="bond1", as_of_date="2026-11-16",
        metric=pb.METRIC_KIND_DURATION_MOD,
        features=pb.BondFeatures(
            years_to_maturity=2, years_to_yield_date=2,
            coupon_type_asof="fixed", coupon_freq_asof=2,
            coupon_period_days_asof=182, coupon_percent=8,
            coupon_value=40, currency_asof="RUB", face_value=1000,
            ytm=12, has_buyback=False, is_government=True,
            settlement_calendar_exact=True, is_structured=False,
        ),
    )


class FakeProvider:
    def explain(self, _facts):
        return {"text": "Итог 800 RUB", "fact_ids": ["total_rub"], "model_version": "test-llm"}


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        make_artifacts(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_generated_serialization_and_metric_inference(self):
        manager = ModelManager(self.root)
        self.assertTrue(manager.metric_ready)
        request = metric_request()
        decoded = pb.EstimateMissingMetricRequest.FromString(request.SerializeToString())
        self.assertTrue(decoded.features.HasField("ytm"))
        response = manager.estimate(decoded)
        self.assertEqual(response.source, pb.METRIC_SOURCE_CATBOOST)
        self.assertTrue(math.isfinite(response.value))
        self.assertEqual(response.model_version, "test-1")
        self.assertTrue(response.review_required)
        self.assertTrue(response.HasField("low"))
        self.assertTrue(response.HasField("high"))
        self.assertLessEqual(response.low, response.value)
        self.assertGreaterEqual(response.high, response.value)

    def test_observed_is_preserved_and_out_of_scope_rejected(self):
        manager = ModelManager(self.root)
        request = metric_request()
        request.observed_value = 3.25
        response = manager.estimate(request)
        self.assertEqual(response.value, 3.25)
        self.assertEqual(response.source, pb.METRIC_SOURCE_OBSERVED)
        request.ClearField("observed_value")
        request.features.is_structured = True
        with self.assertRaises(ModelError) as error:
            manager.estimate(request)
        self.assertEqual(error.exception.code, "FAILED_PRECONDITION")

    def test_scenario_inference_and_range(self):
        manager = ModelManager(self.root)
        self.assertTrue(manager.scenario_ready)
        request = pb.AnalyzeScenarioRequest(
            request_id="req1", calculation_id="calc1", data_date="2026-11-16",
            shift_bp=100, positions=[pb.ScenarioPosition(
                position_id="bond1", duration_mod=2, ytm=12, face_value=1000,
            )],
        )
        response = manager.analyze(request)
        self.assertEqual(response.target_unit, "RUB_FOR_10_FACE_VALUES")
        self.assertEqual(len(response.positions), 1)
        self.assertTrue(math.isfinite(response.positions[0].model_value))
        request.shift_bp = 500
        with self.assertRaises(ModelError) as error:
            manager.analyze(request)
        self.assertEqual(error.exception.code, "OUT_OF_RANGE")

    def test_missing_and_unapproved_artifacts_are_not_ready(self):
        with tempfile.TemporaryDirectory() as empty:
            manager = ModelManager(Path(empty))
            self.assertFalse(manager.metric_ready)
            self.assertFalse(manager.scenario_ready)
            with self.assertRaises(ModelError):
                manager.estimate(metric_request())
            server, port = create_server(manager, None, "127.0.0.1:0")
            server.start()
            try:
                with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
                    stub = pb_grpc.MLServiceStub(channel)
                    readiness = stub.GetReadiness(pb.ReadinessRequest(), timeout=3)
                    self.assertFalse(readiness.metric_ready)
                    self.assertFalse(readiness.scenario_ready)
                    self.assertFalse(readiness.explanation_ready)
                    with self.assertRaises(grpc.RpcError) as error:
                        stub.EstimateMissingMetric(metric_request(), timeout=3)
                    self.assertEqual(error.exception.code(), grpc.StatusCode.FAILED_PRECONDITION)
            finally:
                server.stop(0).wait()
        scenario_path = self.root / "scenario" / "metadata.json"
        metadata = json.loads(scenario_path.read_text(encoding="utf-8"))
        metadata["approved_for_service"] = False
        with tempfile.TemporaryDirectory() as temporary:
            copy = Path(temporary) / "scenario"
            copy.mkdir()
            (copy / "scenario.cbm").write_bytes((self.root / "scenario" / "scenario.cbm").read_bytes())
            _write_json(copy / "metadata.json", metadata)
            self.assertFalse(ModelManager(Path(temporary)).scenario_ready)

    def test_grpc_roundtrip_and_structured_error(self):
        manager = ModelManager(self.root)
        server, port = create_server(manager, FakeProvider(), "127.0.0.1:0")
        server.start()
        try:
            with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
                stub = pb_grpc.MLServiceStub(channel)
                health_status = health_pb2_grpc.HealthStub(channel).Check(health_pb2.HealthCheckRequest(), timeout=3)
                self.assertEqual(health_status.status, health_pb2.HealthCheckResponse.SERVING)
                self.assertTrue(stub.GetReadiness(pb.ReadinessRequest(), timeout=3).metric_ready)
                self.assertEqual(stub.EstimateMissingMetric(metric_request(), timeout=3).source, pb.METRIC_SOURCE_CATBOOST)
                scenario = stub.AnalyzeScenario(pb.AnalyzeScenarioRequest(
                    request_id="req3", calculation_id="calc1", data_date="2026-11-16",
                    shift_bp=100, positions=[pb.ScenarioPosition(
                        position_id="bond1", duration_mod=2, ytm=12, face_value=1000,
                    )],
                ), timeout=3)
                self.assertEqual(scenario.calculation_id, "calc1")
                self.assertEqual(len(scenario.positions), 1)
                explanation = stub.ExplainCalculation(pb.ExplainCalculationRequest(
                    request_id="req2", calculation_id="calc1", calculation_kind=pb.CALCULATION_KIND_DV01,
                    currency="RUB", total_rub="800", data_as_of="2026-11-16",
                    positions=[pb.ExplanationPosition(position_id="bond1", side="buy", value_rub="800")],
                ), timeout=3)
                self.assertTrue(explanation.verified)
                self.assertEqual(explanation.calculation_id, "calc1")
                bad = metric_request()
                bad.features.currency_asof = "USD"
                with self.assertRaises(grpc.RpcError) as error:
                    stub.EstimateMissingMetric(bad, timeout=3)
                self.assertEqual(error.exception.code(), grpc.StatusCode.FAILED_PRECONDITION)
                detail = rpc_status.from_call(error.exception).details[0]
                info = error_details_pb2.ErrorInfo()
                detail.Unpack(info)
                self.assertEqual(info.metadata["request_id"], "req1")
        finally:
            server.stop(0).wait()


if __name__ == "__main__":
    unittest.main()
