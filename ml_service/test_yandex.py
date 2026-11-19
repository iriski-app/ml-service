"""Yandex's HTTP contract and errors are tested without credentials or paid calls."""

import io
import json
import os
import unittest
import urllib.error
import urllib.request
from unittest.mock import MagicMock, patch

from .logic import ServiceError, YandexAIStudioProvider, _NoRedirect, provider_from_env
from .test_logic import DATA

ENV = {
    "LLM_PROVIDER": "yandex",
    "YANDEX_API_KEY": "test-secret-not-a-real-key",
    "YANDEX_FOLDER_ID": "test-folder",
    "YANDEX_MODEL": "deepseek-v4.1-flash",
    "LLM_TIMEOUT_SECONDS": "8",
    "LLM_RETRIES": "1",
    "LLM_REASONING_EFFORT": "none",
}


def response(content=None, **overrides):
    body = {
        "model": "gpt://test-folder/deepseek-v4.1-flash",
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(
            content if content is not None else {"text": "DV01: 800 RUB", "fact_ids": ["total_rub"]},
        )}}],
    }
    body.update(overrides)
    result = MagicMock()
    result.__enter__.return_value.read.return_value = json.dumps(body).encode()
    return result


class YandexProviderTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, ENV, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def assert_code(self, code, action):
        with self.assertRaises(ServiceError) as error:
            action()
        self.assertEqual(error.exception.code, code)
        self.assertNotIn(ENV["YANDEX_API_KEY"], str(error.exception))

    def test_request_auth_model_and_structured_output(self):
        provider = provider_from_env()
        self.assertIsInstance(provider, YandexAIStudioProvider)
        with patch("urllib.request.OpenerDirector.open", return_value=response()) as send:
            result = provider.explain(DATA)
        request = send.call_args.args[0]
        self.assertEqual(request.full_url, "https://ai.api.cloud.yandex.net/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), f"Api-Key {ENV['YANDEX_API_KEY']}")
        self.assertEqual(request.get_header("Openai-project"), "test-folder")
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "gpt://test-folder/deepseek-v4.1-flash")
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["reasoning_effort"], "none")
        self.assertLessEqual(payload["max_tokens"], 1000)
        self.assertNotIn(ENV["YANDEX_API_KEY"], request.data.decode())
        self.assertEqual(result["model_version"], payload["model"])
        self.assertGreater(send.call_args.kwargs["timeout"], 0)
        self.assertLessEqual(send.call_args.kwargs["timeout"], 8)

    def test_missing_configuration_and_invalid_settings(self):
        for key in ("YANDEX_API_KEY", "YANDEX_FOLDER_ID", "YANDEX_MODEL"):
            with self.subTest(key=key), patch.dict(os.environ, {key: ""}):
                self.assert_code("FAILED_PRECONDITION", provider_from_env)
        for values in ({"LLM_PROVIDER": "unknown"}, {"LLM_TIMEOUT_SECONDS": "nan"},
                       {"LLM_TIMEOUT_SECONDS": "0"}, {"LLM_RETRIES": "invalid"},
                       {"YANDEX_FOLDER_ID": "folder/other"}, {"YANDEX_MODEL": "bad\nmodel"},
                       {"LLM_REASONING_EFFORT": "invalid"}):
            with self.subTest(values=values), patch.dict(os.environ, values):
                self.assert_code("FAILED_PRECONDITION", provider_from_env)

    def test_pinned_model_suffix_is_preserved(self):
        with patch.dict(os.environ, {"YANDEX_MODEL": "deepseek-v4.1-flash/1017"}):
            self.assertEqual(provider_from_env().model, "gpt://test-folder/deepseek-v4.1-flash/1017")

    def test_retry_only_transient_http_errors(self):
        for status in (429, 500, 503):
            error = urllib.error.HTTPError("https://example.invalid", status, "redacted", {}, io.BytesIO())
            with self.subTest(status=status), patch("time.sleep"), patch(
                "urllib.request.OpenerDirector.open", side_effect=[error, response()],
            ) as send:
                self.assertEqual(provider_from_env().explain(DATA)["text"], "DV01: 800 RUB")
                self.assertEqual(send.call_count, 2)
        for status in (301, 400, 401, 403):
            error = urllib.error.HTTPError("https://example.invalid", status, "redacted", {}, io.BytesIO())
            with self.subTest(status=status), patch("urllib.request.OpenerDirector.open", side_effect=error) as send:
                self.assert_code("UNAVAILABLE", lambda: provider_from_env().explain(DATA))
                self.assertEqual(send.call_count, 1)

    def test_timeouts_and_retry_limit(self):
        with patch.dict(os.environ, {"LLM_RETRIES": "99"}), patch("time.sleep"), patch(
            "urllib.request.OpenerDirector.open", side_effect=TimeoutError(ENV["YANDEX_API_KEY"]),
        ) as send:
            self.assert_code("UNAVAILABLE", lambda: provider_from_env().explain(DATA))
            self.assertEqual(send.call_count, 3)

    def test_rpc_deadline_limits_attempts(self):
        provider = provider_from_env()
        with patch("urllib.request.OpenerDirector.open", side_effect=TimeoutError()) as send:
            self.assert_code("DEADLINE_EXCEEDED", lambda: provider.explain(DATA, timeout=0.05))
            self.assertEqual(send.call_count, 1)
            self.assertLessEqual(send.call_args.kwargs["timeout"], 0.05)
        with patch("urllib.request.OpenerDirector.open") as send:
            self.assert_code("DEADLINE_EXCEEDED", lambda: provider.explain(DATA, timeout=0))
            send.assert_not_called()

    def test_invalid_provider_responses(self):
        bad = response()
        bad.__enter__.return_value.read.return_value = b"not json"
        huge = response()
        huge.__enter__.return_value.read.return_value = b"x" * (1024 * 1024 + 1)
        for reply in (bad, huge, response(content=[]), response(choices=[]),
                      response(choices=[{"finish_reason": "length", "message": {"content": "{}"}}]),
                      response(model=123)):
            with self.subTest(reply=reply), patch("urllib.request.OpenerDirector.open", return_value=reply):
                self.assert_code("DATA_LOSS", lambda: provider_from_env().explain(DATA))

    def test_redirects_are_not_followed(self):
        request = urllib.request.Request("https://example.invalid", headers={"Authorization": "Api-Key secret"})
        self.assertIsNone(_NoRedirect().redirect_request(request, None, 302, "redirect", {}, "https://other.invalid"))


if __name__ == "__main__":
    unittest.main()
