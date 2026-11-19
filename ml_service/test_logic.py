import unittest
import urllib.error
from unittest.mock import MagicMock, patch

from .logic import ChatCompletionsProvider, ServiceError, explain_calculation

DATA = {
    "calculation_id": "calc1",
    "calculation_kind": "STRESS_CHANGE",
    "currency": "RUB",
    "data_as_of": "2026-11-16T10:00:00Z",
    "total_rub": 800,
    "positions": [
        {"id": "bond1", "side": "BUY", "value_rub": 1000, "change_rub": -20},
        {"id": "bond2", "side": "SELL", "value_rub": -200, "change_rub": 5},
    ],
    "estimated": False,
    "incomplete": False,
}


class FakeProvider:
    def __init__(self, answer=None, error=None):
        self.answer = answer
        self.error = error
        self.seen = None

    def explain(self, facts, *, timeout=None):
        self.seen = facts
        if self.error:
            raise self.error
        return self.answer


class LogicTests(unittest.TestCase):
    def assert_code(self, expected, action):
        with self.assertRaises(ServiceError) as error:
            action()
        self.assertEqual(error.exception.code, expected)

    def test_valid_explanation_and_minimal_facts(self):
        provider = FakeProvider({
            "text": "Итог calc1: 800 RUB. Позиция bond1 изменилась на -20 RUB.",
            "fact_ids": ["total_rub", "position:bond1:change_rub"],
            "model_version": "mock",
        })
        result = explain_calculation(DATA, provider)
        self.assertTrue(result["verified"])
        self.assertEqual(result["calculation_id"], "calc1")
        self.assertEqual(set(provider.seen), {"calculation_id", "calculation_kind", "currency", "data_as_of", "incomplete", "estimated", "total_rub", "positions", "facts", "fact_units", "calculation_details_available", "explanation_scope", "limitations"})

    def test_wrong_number_and_reference(self):
        self.assert_code("DATA_LOSS", lambda: explain_calculation(DATA, FakeProvider({"text": "Итог 801 RUB", "fact_ids": ["total_rub"]})))
        self.assert_code("DATA_LOSS", lambda: explain_calculation(DATA, FakeProvider({"text": "Итог 800 RUB", "fact_ids": ["missing"]})))
        self.assert_code("DATA_LOSS", lambda: explain_calculation(DATA, FakeProvider({"text": "Позиция bond2; итог 800 RUB", "fact_ids": ["total_rub"]})))

    def test_timeout_and_missing_provider(self):
        self.assert_code("UNAVAILABLE", lambda: explain_calculation(DATA, FakeProvider(error=TimeoutError())))
        self.assert_code("FAILED_PRECONDITION", lambda: explain_calculation(DATA, None))

    def test_inconsistent_total(self):
        bad = {**DATA, "total_rub": 801}
        self.assert_code("INVALID_ARGUMENT", lambda: explain_calculation(bad, FakeProvider({})))

    def test_dv01_facts_need_no_scenario_change(self):
        data = {**DATA, "calculation_kind": "DV01", "positions": [
            {"id": "bond1", "side": "BUY", "value_rub": 800},
        ]}
        provider = FakeProvider({"text": "DV01: 800 RUB", "fact_ids": ["total_rub"]})
        self.assertTrue(explain_calculation(data, provider)["verified"])

    def test_dv01_summary_must_identify_the_risk_metric(self):
        data = {**DATA, "calculation_kind": "DV01", "positions": [
            {"id": "bond1", "side": "BUY", "value_rub": 800},
        ]}
        self.assert_code("DATA_LOSS", lambda: explain_calculation(data, FakeProvider({
            "text": "Стоимость позиции составляет 800 RUB", "fact_ids": ["total_rub"],
        })))

    def test_provider_retries_network_failure_once(self):
        provider = ChatCompletionsProvider("https://example.invalid/chat", "test-model", "test-key", timeout=1, retries=1)
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"choices":[{"message":{"content":"{\\"text\\":\\"800 RUB\\",\\"fact_ids\\":[\\"total_rub\\"]}"}}],"model":"test-model"}'
        with (
            patch("urllib.request.OpenerDirector.open", side_effect=[urllib.error.URLError("offline"), response]) as urlopen,
            patch("time.sleep"),
        ):
            result = provider.explain({"facts": {"total_rub": "800"}})
        self.assertEqual(result["model_version"], "test-model")
        self.assertEqual(urlopen.call_count, 2)

    def test_numeric_claims_cannot_hide_in_identifiers_or_units(self):
        for text in ("Итог 999руб.", "Итог 8e3 RUB", "Итог 8 000 RUB", "Итог bond1800 RUB", "Итог −800 RUB"):
            with self.subTest(text=text):
                self.assert_code("DATA_LOSS", lambda text=text: explain_calculation(DATA, FakeProvider({
                    "text": text, "fact_ids": ["total_rub"],
                })))
        data = {**DATA, "calculation_id": "999"}
        self.assert_code("DATA_LOSS", lambda: explain_calculation(data, FakeProvider({
            "text": "Итог 999 RUB", "fact_ids": ["total_rub"],
        })))

    def test_decimal_comma_and_unicode_minus(self):
        provider = FakeProvider({"text": "Изменение bond1: −20,0 RUB", "fact_ids": ["position:bond1:change_rub"]})
        self.assertTrue(explain_calculation(DATA, provider)["verified"])

    def test_malformed_output_and_duplicate_references(self):
        for output in (None, [], {"text": "800 RUB", "fact_ids": ["total_rub", "total_rub"]}):
            self.assert_code("DATA_LOSS", lambda output=output: explain_calculation(DATA, FakeProvider(output)))

    def test_quality_flags_and_extra_input_data(self):
        positions = [dict(p) for p in DATA["positions"]]
        positions[0]["estimated"] = True
        provider = FakeProvider({"text": "Итог оценочный: 800 RUB", "fact_ids": ["total_rub"]})
        explain_calculation({**DATA, "positions": positions, "session_token": "private"}, provider)
        self.assertTrue(provider.seen["estimated"])
        self.assertNotIn("session_token", provider.seen)
        positions[0]["estimated"] = "false"
        self.assert_code("INVALID_ARGUMENT", lambda: explain_calculation({**DATA, "positions": positions}, provider))


if __name__ == "__main__":
    unittest.main()
