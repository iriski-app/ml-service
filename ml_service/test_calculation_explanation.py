"""Check explanations against explicit, hand-calculated examples from the methodology."""

import copy
import tempfile
import unittest
from pathlib import Path

import grpc

from .logic import ServiceError, explain_calculation
from .models import ModelManager
from .server import create_server
from .test_logic import FakeProvider
from .v1 import service_pb2 as pb
from .v1 import service_pb2_grpc as rpc


def calculation():
    return {
        "calculation_id": "calc-method", "calculation_kind": "DV01", "currency": "RUB",
        "data_as_of": "2026-11-17", "total_rub": "-199.97",
        "method": "duration_convexity", "yield_shift_bp": 1,
        "positions": [{
            "id": "bond1", "isin": "RU000A0JX0J2", "side": "BUY", "value_rub": "-199.97",
            "inputs": {
                "current_fair_value": "1000000", "position_currency": "RUB", "fx_rate_to_rub": "1",
                "duration_mod": 2, "convexity": 6, "duration_source": "observed",
                "convexity_source": "observed", "coupon_type": "fixed",
            },
        }],
    }


class ExplanationProvider:
    def __init__(self):
        self.seen = None

    def explain(self, source, *, timeout=None):
        self.seen = source
        return {
            "text": f"DV01/стресс: {source['total_rub']} RUB. Применены стоимость, дюрация и выпуклость при заданном сдвиге доходности.",
            "fact_ids": ["total_rub", "scenario:yield_shift_bp", "position:bond1:current_fair_value",
                         "position:bond1:duration_mod", "position:bond1:convexity"],
        }


class CalculationExplanationTests(unittest.TestCase):
    def assert_invalid(self, data):
        provider = ExplanationProvider()
        with self.assertRaises(ServiceError) as error:
            explain_calculation(data, provider)
        self.assertEqual(error.exception.code, "INVALID_ARGUMENT")
        self.assertIsNone(provider.seen)

    def test_dv01_details_and_units(self):
        provider = ExplanationProvider()
        answer = explain_calculation(calculation(), provider)
        self.assertTrue(answer["calculation_details_available"])
        self.assertEqual(provider.seen["facts"]["scenario:yield_shift_fraction"], "0.0001")
        self.assertEqual(provider.seen["facts"]["scenario:yield_shift_percentage_points"], "0.01")
        self.assertEqual(provider.seen["fact_units"]["position:bond1:duration_mod"], "years")
        self.assertEqual(provider.seen["positions"][0]["isin"], "RU000A0JX0J2")
        self.assertTrue(provider.seen["methodology"]["is_approximation"])

    def test_zero_positive_negative_shift_and_trade_direction(self):
        # One million RUB, D=2, C=6. Values are independent hand calculations.
        for shift, side, expected in [(0, "BUY", "0"), (100, "BUY", "-19700"),
                                       (-100, "BUY", "20300"), (100, "SELL", "19700"),
                                       (-100, "SELL", "-20300")]:
            with self.subTest(shift=shift, side=side):
                data = calculation()
                data.update(calculation_kind="STRESS_CHANGE", yield_shift_bp=shift, total_rub=expected)
                data["positions"][0].update(side=side, value_rub=expected, change_rub=expected)
                provider = ExplanationProvider()
                self.assertTrue(explain_calculation(data, provider)["verified"])
                self.assertEqual(provider.seen["facts"]["scenario:yield_shift_bp"], str(shift))

    def test_foreign_currency_uses_original_fx_rate(self):
        data = calculation()
        data["positions"][0]["inputs"].update(current_fair_value="10000", position_currency="USD", fx_rate_to_rub="100")
        provider = ExplanationProvider()
        self.assertTrue(explain_calculation(data, provider)["verified"])
        self.assertEqual(provider.seen["fact_units"]["position:bond1:current_fair_value"], "USD")
        bad = copy.deepcopy(data)
        bad["positions"][0]["inputs"]["fx_rate_to_rub"] = "90"
        self.assert_invalid(bad)

    def test_catboost_origin_is_explained_as_estimate(self):
        data = calculation()
        data["positions"][0]["inputs"]["duration_source"] = "catboost"
        provider = ExplanationProvider()
        explain_calculation(data, provider)
        self.assertTrue(provider.seen["estimated"])
        self.assertTrue(provider.seen["positions"][0]["estimated"])
        self.assertEqual(provider.seen["positions"][0]["inputs"]["duration_source"], "catboost")

    def test_floating_coupon_uses_case_rule(self):
        data = calculation()
        data["total_rub"] = "-50"
        data["positions"][0]["value_rub"] = "-50"
        data["positions"][0]["inputs"].update(
            coupon_type="floating", coupon_frequency=2, duration_mod=0.5, convexity=0,
            duration_source="coupon_rule", convexity_source="coupon_rule",
        )
        self.assertTrue(explain_calculation(data, ExplanationProvider())["verified"])
        for changes in ({"duration_mod": 2}, {"convexity": 1}, {"coupon_frequency": 0},
                        {"duration_source": "catboost"}):
            bad = copy.deepcopy(data)
            bad["positions"][0]["inputs"].update(changes)
            self.assert_invalid(bad)

    def test_inconsistent_or_incomplete_inputs_are_rejected(self):
        for changes in ({"duration_mod": 3}, {"convexity": float("nan")},
                        {"current_fair_value": "-1000000"}, {"fx_rate_to_rub": "0"},
                        {"duration_source": "unknown"}, {"coupon_type": "unknown"}):
            data = calculation()
            data["positions"][0]["inputs"].update(changes)
            self.assert_invalid(data)
        for field in ("inputs",):
            data = calculation()
            del data["positions"][0][field]
            self.assert_invalid(data)
        data = calculation()
        data["yield_shift_bp"] = 100
        self.assert_invalid(data)
        data = calculation()
        del data["method"]
        self.assert_invalid(data)

    def test_details_cannot_be_replaced_by_a_summary_without_input_references(self):
        with self.assertRaises(ServiceError) as error:
            explain_calculation(calculation(), FakeProvider({
                "text": "DV01: -199.97 RUB", "fact_ids": ["total_rub"],
            }))
        self.assertEqual(error.exception.code, "DATA_LOSS")

    def test_unconfirmed_duration_in_text_is_rejected(self):
        provider = ExplanationProvider()
        answer = provider.explain({"total_rub": "-199.97"})
        answer["text"] = "DV01: -199.97 RUB. Дюрация 3 года."
        with self.assertRaises(ServiceError) as error:
            explain_calculation(calculation(), FakeProvider(answer))
        self.assertEqual(error.exception.code, "DATA_LOSS")

    def test_legacy_call_is_explicitly_result_only(self):
        data = calculation()
        data.pop("method")
        data.pop("yield_shift_bp")
        data["positions"][0].pop("inputs")
        provider = FakeProvider({"text": "DV01: -199.97 RUB", "fact_ids": ["total_rub"]})
        answer = explain_calculation(data, provider)
        self.assertFalse(answer["calculation_details_available"])
        self.assertTrue(answer["limitations"])
        self.assertEqual(provider.seen["explanation_scope"], "result_only")

    def test_full_grpc_request_preserves_inputs_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = ExplanationProvider()
            server, port = create_server(ModelManager(Path(directory)), provider, "127.0.0.1:0")
            server.start()
            try:
                with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
                    answer = rpc.MLServiceStub(channel).ExplainCalculation(pb.ExplainCalculationRequest(
                        request_id="details-1", calculation_id="calc-method",
                        calculation_kind=pb.CALCULATION_KIND_DV01, currency="RUB", total_rub="-199.97",
                        data_as_of="2026-11-17", yield_shift_bp=1,
                        method=pb.EXPLANATION_METHOD_DURATION_CONVEXITY,
                        positions=[pb.ExplanationPosition(
                            position_id="bond1", isin="RU000A0JX0J2", side="buy", value_rub="-199.97",
                            inputs=pb.PositionCalculationInputs(
                                current_fair_value="1000000", position_currency="RUB", fx_rate_to_rub="1",
                                duration_mod=2, convexity=6, coupon_type="fixed",
                                duration_source=pb.METRIC_SOURCE_CATBOOST, convexity_source=pb.METRIC_SOURCE_OBSERVED,
                            ),
                        )],
                    ), timeout=3)
                self.assertTrue(answer.verified)
                self.assertTrue(answer.calculation_details_available)
                self.assertEqual(answer.calculation_id, "calc-method")
                self.assertTrue(provider.seen["estimated"])
                self.assertEqual(provider.seen["positions"][0]["inputs"]["convexity_source"], "observed")
            finally:
                server.stop(0).wait()


if __name__ == "__main__":
    unittest.main()
