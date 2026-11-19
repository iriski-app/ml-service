"""Exercise the scenario training/export path and its gRPC serving contract."""

import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import grpc
import pandas as pd
from catboost import CatBoostError

from scripts.prepare_scenario_bundle import package
from scripts.train_scenario_model import scenarios, train

from .models import ModelManager
from .server import create_server
from .v1 import service_pb2 as pb
from .v1 import service_pb2_grpc as rpc


class ScenarioBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.directory = cls.root / "scenario"
        data = pd.DataFrame([
            {"SECID": f"bond{i}", "TRADEDATE": f"2025-01-{day:02d}", "MATDATE": "2030-01-01",
             "CLOSE": 100, "DURATION": 365 + 30 * i, "YIELDCLOSE": 10 + i / 2, "FACEVALUE": 1000 + 10 * i}
            for i in range(20) for day in range(1, 4)
        ])
        dataset = cls.root / "dataset.csv"
        data.to_csv(dataset, index=False)
        cls.report = train(dataset, cls.directory, iterations=12)
        package(cls.directory / "scenario.cbm", cls.directory, "scenario-test-v1",
                validation_report=cls.directory / "training_report.json", enable=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def request(self, shift=100):
        ranges = self.report["input_ranges"]
        position = pb.ScenarioPosition(position_id="position1", **{
            name: (low + high) / 2 for name, (low, high) in ranges.items()
        })
        return pb.AnalyzeScenarioRequest(request_id="scenario-request", calculation_id="calculation1",
                                         data_date="2026-11-19", shift_bp=shift, positions=[position])

    def test_independent_splits_and_known_target_units(self):
        splits = [set(self.report["splits"][name]["bonds"]) for name in ("train", "validation", "test")]
        self.assertEqual(len(set.union(*splits)), 20)
        self.assertFalse(splits[0] & splits[1] or splits[0] & splits[2] or splits[1] & splits[2])
        self.assertTrue(math.isfinite(self.report["test_mae_rub_for_10_face_values"]))
        frame = scenarios(pd.DataFrame([{"DurationMod": 2, "Convexity": 6, "FACEVALUE": 1000}]))
        changes = frame.set_index("shift_bp")["target"]
        self.assertAlmostEqual(changes[100], -197)
        self.assertAlmostEqual(changes[-100], 203)
        self.assertEqual(changes[0], 0)

    def test_real_bundle_and_grpc_for_both_directions_and_zero(self):
        manager = ModelManager(self.root)
        self.assertTrue(manager.scenario_ready, manager.scenario_error)
        server, port = create_server(manager, None, "127.0.0.1:0")
        server.start()
        try:
            with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
                stub = rpc.MLServiceStub(channel)
                self.assertTrue(stub.GetReadiness(pb.ReadinessRequest(), timeout=3).scenario_ready)
                for shift in [-200, -100, 0, 100, 200]:
                    response = stub.AnalyzeScenario(self.request(shift), timeout=3)
                    self.assertEqual(response.request_id, "scenario-request")
                    self.assertEqual(response.calculation_id, "calculation1")
                    self.assertEqual(response.target_kind, "formula_proxy_reference_position")
                    self.assertEqual(response.target_unit, "RUB_FOR_10_FACE_VALUES")
                    self.assertEqual(response.model_version, "scenario-test-v1")
                    self.assertTrue(response.experimental)
                    self.assertTrue(response.limitations)
                    self.assertEqual(response.positions[0].position_id, "position1")
                    self.assertTrue(math.isfinite(response.positions[0].model_value))
                    if shift == 0:
                        self.assertEqual(response.positions[0].model_value, 0)
                interpolated = stub.AnalyzeScenario(self.request(75), timeout=3)
                self.assertTrue(any("interpolation" in message for message in interpolated.limitations))
        finally:
            server.stop(0).wait()

    def test_invalid_inputs_ranges_and_missing_fields(self):
        manager = ModelManager(self.root)
        server, port = create_server(manager, None, "127.0.0.1:0")
        server.start()
        try:
            with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
                call = rpc.MLServiceStub(channel).AnalyzeScenario
                for shift in [-201, 201, float("nan"), float("inf")]:
                    with self.subTest(shift=shift), self.assertRaises(grpc.RpcError) as error:
                        call(self.request(shift), timeout=3)
                    self.assertEqual(error.exception.code(), grpc.StatusCode.OUT_OF_RANGE)
                for name in ["duration_mod", "ytm", "face_value"]:
                    request = self.request()
                    request.positions[0].ClearField(name)
                    with self.assertRaises(grpc.RpcError) as error:
                        call(request, timeout=3)
                    self.assertEqual(error.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)
                    request = self.request()
                    setattr(request.positions[0], name, self.report["input_ranges"][name][1] + 1)
                    with self.assertRaises(grpc.RpcError) as error:
                        call(request, timeout=3)
                    self.assertEqual(error.exception.code(), grpc.StatusCode.OUT_OF_RANGE)
                request = self.request()
                request.positions.add().CopyFrom(request.positions[0])
                with self.assertRaises(grpc.RpcError) as error:
                    call(request, timeout=3)
                self.assertEqual(error.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)
                request.ClearField("positions")
                with self.assertRaises(grpc.RpcError) as error:
                    call(request, timeout=3)
                self.assertEqual(error.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)
        finally:
            server.stop(0).wait()

    def test_corrupt_artifact_report_and_wrong_target_are_not_ready(self):
        for name in ["scenario.cbm", "training_report.json", "target", "domain"]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                folder = root / "scenario"
                shutil.copytree(self.directory, folder)
                if name in ("target", "domain"):
                    path = folder / "metadata.json"
                    metadata = json.loads(path.read_text())
                    if name == "target":
                        metadata["target_unit"] = "RUB_ACTUAL_BASKET"
                    else:
                        metadata["input_ranges"]["duration_mod"] = [float("nan"), 10]
                    path.write_text(json.dumps(metadata))
                else:
                    with (folder / name).open("ab") as output:
                        output.write(b"corrupted")
                self.assertFalse(ModelManager(root).scenario_ready)

    def test_packaging_never_enables_without_disjoint_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package(self.directory / "scenario.cbm", root / "scenario", "disabled-v1")
            self.assertFalse(ModelManager(root).scenario_ready)
            with self.assertRaises(ValueError):
                package(self.directory / "scenario.cbm", root / "scenario", "bad-v1", enable=True)
            report = json.loads(json.dumps(self.report))
            report["splits"]["test"]["bonds"] = report["splits"]["train"]["bonds"]
            path = root / "bad-report.json"
            path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                package(self.directory / "scenario.cbm", root / "scenario", "bad-v1",
                        validation_report=path, enable=True)
            report = json.loads(json.dumps(self.report))
            report["model_sha256"] = "0" * 64
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "does not belong"):
                package(self.directory / "scenario.cbm", root / "scenario", "bad-v1",
                        validation_report=path, enable=True)

    def test_inference_failure_is_a_controlled_grpc_error(self):
        manager = ModelManager(self.root)
        server, port = create_server(manager, None, "127.0.0.1:0")
        server.start()
        try:
            with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
                with (patch.object(manager.scenario_model, "predict", side_effect=CatBoostError("failed")),
                      self.assertRaises(grpc.RpcError) as error):
                    rpc.MLServiceStub(channel).AnalyzeScenario(self.request(), timeout=3)
                self.assertEqual(error.exception.code(), grpc.StatusCode.UNAVAILABLE)
        finally:
            server.stop(0).wait()

    def test_training_rejects_duplicate_data_and_too_few_bonds(self):
        source = pd.read_csv(self.root / "dataset.csv")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            duplicate = root / "duplicate.csv"
            pd.concat([source, source.iloc[:1]]).to_csv(duplicate, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                train(duplicate, root / "out", iterations=1)
            small = root / "small.csv"
            source.iloc[:6].to_csv(small, index=False)
            with self.assertRaisesRegex(ValueError, "ten"):
                train(small, root / "out", iterations=1)
