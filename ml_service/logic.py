"""Provider-independent logic for the two MLService methods."""

from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol


class ServiceError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def number(value: Any, field: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ServiceError("INVALID_ARGUMENT", f"invalid {field}")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ServiceError("INVALID_ARGUMENT", f"invalid {field}") from exc
    if not result.is_finite():
        raise ServiceError("INVALID_ARGUMENT", f"invalid {field}")
    return result


class Provider(Protocol):
    def explain(self, facts: dict[str, Any]) -> dict[str, Any]: ...


class ChatCompletionsProvider:
    """OpenAI-compatible chat completions endpoint configured entirely via env."""

    def __init__(self, url: str, model: str, api_key: str, timeout: float = 8.0, retries: int = 1):
        if not url.startswith("https://") or not model or not api_key or not math.isfinite(timeout) or timeout <= 0:
            raise ServiceError("FAILED_PRECONDITION", "LLM provider is not configured")
        self.url, self.model, self.api_key = url, model, api_key
        self.timeout, self.retries = timeout, max(0, min(retries, 2))

    @classmethod
    def from_env(cls) -> ChatCompletionsProvider:
        try:
            timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "8"))
            retries = int(os.getenv("LLM_RETRIES", "1"))
        except ValueError as exc:
            raise ServiceError("FAILED_PRECONDITION", "invalid LLM provider settings") from exc
        return cls(os.getenv("LLM_API_URL", ""), os.getenv("LLM_MODEL", ""), os.getenv("LLM_API_KEY", ""), timeout, retries)

    def explain(self, facts: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": (
                    "Summarize the supplied bond risk calculation briefly in Russian. "
                    "Return only a JSON object with keys text and fact_ids. "
                    "Every numeric statement must be copied exactly from a cited fact. "
                    "Do not invent financial data or cite unknown fact IDs."
                )},
                {"role": "user", "content": json.dumps(facts, ensure_ascii=False, separators=(",", ":"))},
            ],
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.load(response)
                content = body["choices"][0]["message"]["content"]
                result = json.loads(content)
                if not isinstance(result, dict):
                    raise TypeError("model output is not an object")
                result["model_version"] = body.get("model", self.model)
                return result
            except urllib.error.HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if retryable and attempt < self.retries:
                    time.sleep(0.2 * (attempt + 1))
                    continue
                raise ServiceError("UNAVAILABLE", "LLM provider error") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt < self.retries:
                    time.sleep(0.2 * (attempt + 1))
                    continue
                raise ServiceError("UNAVAILABLE", "LLM provider unavailable") from exc
            except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ServiceError("DATA_LOSS", "invalid LLM provider response") from exc
        raise ServiceError("UNAVAILABLE", "LLM provider unavailable")


_IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
_NUMERIC_TOKEN = re.compile(r"(?<![\w.])[-+]?\d+(?:[.,]\d+)?(?![\w.])")


def build_facts(data: dict[str, Any]) -> dict[str, Any]:
    calculation_id = data.get("calculation_id")
    if not isinstance(calculation_id, str) or not _IDENTIFIER.fullmatch(calculation_id):
        raise ServiceError("INVALID_ARGUMENT", "invalid calculation_id")
    if data.get("currency") != "RUB" or not isinstance(data.get("data_as_of"), str) or not data["data_as_of"]:
        raise ServiceError("INVALID_ARGUMENT", "currency RUB and data_as_of are required")
    if data.get("calculation_kind") not in ("DV01", "STRESS_CHANGE"):
        raise ServiceError("INVALID_ARGUMENT", "invalid calculation kind")
    try:
        datetime.fromisoformat(data["data_as_of"])
    except ValueError as exc:
        raise ServiceError("INVALID_ARGUMENT", "invalid data_as_of") from exc
    if not isinstance(data.get("incomplete", False), bool) or not isinstance(data.get("estimated", False), bool):
        raise ServiceError("INVALID_ARGUMENT", "invalid quality flags")
    positions = data.get("positions")
    if not isinstance(positions, list) or not positions or len(positions) > 1000:
        raise ServiceError("INVALID_ARGUMENT", "positions are required")
    total = number(data.get("total_rub"), "total_rub")
    facts: dict[str, str] = {"total_rub": str(total)}
    slim_positions: list[dict[str, Any]] = []
    seen: set[str] = set()
    position_total = Decimal(0)
    for p in positions:
        if not isinstance(p, dict) or not isinstance(p.get("id"), str) or not _IDENTIFIER.fullmatch(p["id"]) or p["id"] in seen or p.get("side") not in ("BUY", "SELL"):
            raise ServiceError("INVALID_ARGUMENT", "invalid position")
        seen.add(p["id"])
        value = number(p.get("value_rub"), "value_rub")
        change = number(p["change_rub"], "change_rub") if p.get("change_rub") not in (None, "") else None
        if data["calculation_kind"] == "STRESS_CHANGE" and change is None:
            raise ServiceError("INVALID_ARGUMENT", "stress change is required")
        position_total += value
        facts[f"position:{p['id']}:value_rub"] = str(value)
        if change is not None:
            facts[f"position:{p['id']}:change_rub"] = str(change)
        slim_positions.append({"id": p["id"], "side": p["side"], "value_rub": str(value), "change_rub": str(change) if change is not None else None, "estimated": bool(p.get("estimated", False))})
    if position_total != total:
        raise ServiceError("INVALID_ARGUMENT", "total_rub does not match positions")
    return {
        "calculation_id": calculation_id,
        "calculation_kind": data["calculation_kind"],
        "currency": "RUB",
        "data_as_of": data["data_as_of"],
        "incomplete": bool(data.get("incomplete", False)),
        "estimated": bool(data.get("estimated", False)),
        "total_rub": str(total),
        "positions": slim_positions,
        "facts": facts,
    }


def validate_explanation(source: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    text = output.get("text")
    references = output.get("fact_ids")
    if not isinstance(text, str) or not text.strip() or len(text) > 2000 or not isinstance(references, list) or not references or not all(isinstance(x, str) and x in source["facts"] for x in references):
        raise ServiceError("DATA_LOSS", "unverified explanation")
    if len(set(references)) != len(references):
        raise ServiceError("DATA_LOSS", "duplicate fact references")
    for position in source["positions"]:
        if position["id"] in text and not any(ref.startswith(f"position:{position['id']}:") for ref in references):
            raise ServiceError("DATA_LOSS", "uncited position")
    # Identifiers and the data date are not financial claims. Mask them before
    # checking all numeric tokens against values of explicitly cited facts.
    checked_text = text.replace(source["calculation_id"], "")
    checked_text = checked_text.replace(source["data_as_of"], "")
    for position in source["positions"]:
        checked_text = checked_text.replace(position["id"], "")
    allowed = {Decimal(source["facts"][ref]) for ref in references}
    for match in _NUMERIC_TOKEN.finditer(checked_text):
        try:
            observed = Decimal(match.group().replace(",", "."))
        except InvalidOperation as exc:
            raise ServiceError("DATA_LOSS", "unverified numeric claim") from exc
        if observed not in allowed:
            raise ServiceError("DATA_LOSS", "unverified numeric claim")
    return {
        "calculation_id": source["calculation_id"],
        "text": text,
        "fact_ids": references,
        "model_version": str(output.get("model_version", "unknown")),
        "verified": True,
    }


def explain_calculation(data: dict[str, Any], provider: Provider | None) -> dict[str, Any]:
    source = build_facts(data)
    if provider is None:
        raise ServiceError("FAILED_PRECONDITION", "LLM provider is not configured")
    try:
        output = provider.explain(source)
    except ServiceError:
        raise
    except Exception as exc:
        raise ServiceError("UNAVAILABLE", "LLM provider unavailable") from exc
    return validate_explanation(source, output)
