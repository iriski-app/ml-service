"""Load enabled CatBoost artifacts once and perform inference only."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostError, CatBoostRegressor

from ml_service.v1 import service_pb2 as pb

IMPUTATION_FEATURES = [
    "years_to_maturity", "years_to_yield_date", "coupon_type_asof",
    "coupon_freq_asof", "coupon_period_days_asof", "coupon_percent",
    "coupon_value", "currency_asof", "face_value", "ytm",
    "has_buyback", "is_government",
]
IMPUTATION_CATEGORICAL = ["coupon_type_asof", "currency_asof"]
SCENARIO_FEATURES = ["shock_abs", "DurationMod", "YIELDCLOSE", "FACEVALUE"]
METRIC_NAMES = {
    pb.METRIC_KIND_DURATION_MOD: "duration_mod",
    pb.METRIC_KIND_CONVEXITY: "convexity",
}


class ModelError(Exception):
    def __init__(self, code: str, reason: str):
        super().__init__(reason)
        self.code = code
        self.reason = reason


def finite(value: float) -> bool:
    return math.isfinite(value)


def _json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("metadata must be an object")
    return data


def _model(path: Path, checksums: dict[str, str]) -> CatBoostRegressor:
    expected = checksums.get(path.name)
    if not expected or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"missing or incorrect checksum for {path.name}")
    model = CatBoostRegressor()
    model.load_model(str(path))
    return model


def _optional_number(message: Any, name: str) -> float:
    if not message.HasField(name):
        return math.nan
    value = getattr(message, name)
    if not finite(value):
        raise ModelError("INVALID_ARGUMENT", f"invalid {name}")
    return value


class ModelManager:
    def __init__(self, root: Path):
        self.root = root
        self.metric_models: dict[str, CatBoostRegressor] = {}
        self.interval_models: dict[str, CatBoostRegressor] = {}
        self.metric_metadata: dict[str, Any] = {}
        self.metric_deployment: dict[str, Any] = {}
        self.scenario_model: CatBoostRegressor | None = None
        self.scenario_metadata: dict[str, Any] = {}
        self.metric_error = "imputation artifacts unavailable"
        self.scenario_error = "enabled scenario artifact unavailable"
        self._load_metric()
        self._load_scenario()

    @property
    def metric_ready(self) -> bool:
        return len(self.metric_models) == 2

    @property
    def scenario_ready(self) -> bool:
        return self.scenario_model is not None

    def _load_metric(self) -> None:
        directory = self.root / "imputation"
        try:
            metadata = _json(directory / "metadata.json")
            deployment = _json(directory / "deployment.json")
            if metadata.get("features") != IMPUTATION_FEATURES or metadata.get("categorical") != IMPUTATION_CATEGORICAL or metadata.get("target_transform") != "log1p":
                raise ValueError("imputation preprocessing does not match training")
            if not deployment.get("model_version") or not isinstance(deployment.get("supported_currencies"), list) or not deployment["supported_currencies"] or not finite(float(deployment["years_to_yield_date_min"])) or not finite(float(deployment["years_to_yield_date_max"])):
                raise ValueError("incomplete imputation deployment metadata")
            if float(deployment["years_to_yield_date_min"]) >= float(deployment["years_to_yield_date_max"]):
                raise ValueError("invalid imputation maturity range")
            checksums = deployment["sha256"]
            models = {name: _model(directory / metadata["model_files"][name], checksums) for name in ("duration_mod", "convexity")}
            intervals = {}
            for name in ("duration_mod", "convexity"):
                filename = metadata.get("interval_model_files", {}).get(name)
                if filename and (directory / filename).is_file():
                    intervals[name] = _model(directory / filename, checksums)
            self.metric_models = models
            self.interval_models = intervals
            self.metric_metadata = metadata
            self.metric_deployment = deployment
            self.metric_error = ""
        except (OSError, ValueError, TypeError, KeyError, AttributeError, CatBoostError) as exc:
            self.metric_error = f"imputation artifacts unavailable: {type(exc).__name__}"

    def _load_scenario(self) -> None:
        directory = self.root / "scenario"
        try:
            metadata = _json(directory / "metadata.json")
            enabled = metadata.get("enabled_for_service", metadata.get("approved_for_service", False)) is True
            if metadata.get("features") != SCENARIO_FEATURES or not enabled:
                raise ValueError("scenario model is not enabled for service")
            if (not metadata.get("model_version")
                    or metadata.get("target_kind") != "formula_proxy_reference_position"
                    or metadata.get("target_unit") != "RUB_FOR_10_FACE_VALUES"):
                raise ValueError("scenario target is not defined")
            lower, upper = float(metadata["shift_bp_min"]), float(metadata["shift_bp_max"])
            if not finite(lower) or not finite(upper) or lower >= upper:
                raise ValueError("invalid scenario range")
            ranges = metadata.get("input_ranges", {})
            if metadata.get("enabled_for_service") and set(ranges) != {"duration_mod", "ytm", "face_value"}:
                raise ValueError("scenario training domain is missing")
            for name, bounds in ranges.items():
                if name not in ("duration_mod", "ytm", "face_value") or len(bounds) != 2:
                    raise ValueError("invalid scenario input range")
                low, high = map(float, bounds)
                if not finite(low) or not finite(high) or low > high:
                    raise ValueError("invalid scenario input range")
            report = metadata.get("training_report_file")
            if metadata.get("enabled_for_service"):
                if not report or Path(report).name != report:
                    raise ValueError("scenario validation report is missing")
                expected = metadata["sha256"].get(report)
                if not expected or hashlib.sha256((directory / report).read_bytes()).hexdigest() != expected:
                    raise ValueError("scenario validation report checksum mismatch")
                validation = _json(directory / report)
                if (validation.get("model_sha256") != metadata["sha256"].get(metadata["model_file"])
                        or validation.get("input_ranges") != ranges
                        or validation.get("features") != SCENARIO_FEATURES
                        or validation.get("tested_shifts_bp") != metadata.get("tested_shifts_bp")):
                    raise ValueError("scenario model, domain and validation report do not match")
            model = _model(directory / metadata["model_file"], metadata["sha256"])
            if model.feature_names_ != SCENARIO_FEATURES:
                raise ValueError("scenario artifact features do not match metadata")
            self.scenario_model = model
            self.scenario_metadata = metadata
            self.scenario_error = ""
        except (OSError, ValueError, TypeError, KeyError, AttributeError, CatBoostError) as exc:
            self.scenario_error = f"enabled scenario artifact unavailable: {type(exc).__name__}"

    def estimate(self, request: pb.EstimateMissingMetricRequest) -> pb.EstimateMissingMetricResponse:
        name = METRIC_NAMES.get(request.metric)
        if name is None:
            raise ModelError("INVALID_ARGUMENT", "unknown metric")
        response = pb.EstimateMissingMetricResponse(
            request_id=request.request_id, instrument_id=request.instrument_id,
            as_of_date=request.as_of_date, metric=request.metric,
        )
        if request.HasField("observed_value"):
            if not finite(request.observed_value) or request.observed_value < 0:
                raise ModelError("INVALID_ARGUMENT", "invalid observed value")
            response.value = request.observed_value
            response.source = pb.METRIC_SOURCE_OBSERVED
            return response
        if not self.metric_ready:
            raise ModelError("FAILED_PRECONDITION", self.metric_error)
        if not request.HasField("features"):
            raise ModelError("INVALID_ARGUMENT", "bond features are required")
        features = request.features
        deployment = self.metric_deployment
        if not features.settlement_calendar_exact or not features.HasField("is_structured") or features.is_structured or features.coupon_type_asof not in ("fixed", "zero"):
            raise ModelError("FAILED_PRECONDITION", "bond is outside the supported sample")
        if features.currency_asof not in deployment["supported_currencies"]:
            raise ModelError("FAILED_PRECONDITION", "currency is outside the supported sample")
        for field in ("years_to_maturity", "years_to_yield_date", "face_value"):
            if not features.HasField(field) or not finite(getattr(features, field)) or getattr(features, field) <= 0:
                raise ModelError("INVALID_ARGUMENT", f"{field} is required and must be positive")
        if not float(deployment["years_to_yield_date_min"]) <= features.years_to_yield_date <= float(deployment["years_to_yield_date_max"]):
            raise ModelError("FAILED_PRECONDITION", "maturity is outside the training range")
        row = {
            "years_to_maturity": features.years_to_maturity,
            "years_to_yield_date": features.years_to_yield_date,
            "coupon_type_asof": features.coupon_type_asof,
            "coupon_freq_asof": _optional_number(features, "coupon_freq_asof"),
            "coupon_period_days_asof": _optional_number(features, "coupon_period_days_asof"),
            "coupon_percent": _optional_number(features, "coupon_percent"),
            "coupon_value": _optional_number(features, "coupon_value"),
            "currency_asof": features.currency_asof,
            "face_value": features.face_value,
            "ytm": _optional_number(features, "ytm"),
            "has_buyback": float(features.has_buyback) if features.HasField("has_buyback") else math.nan,
            "is_government": float(features.is_government) if features.HasField("is_government") else math.nan,
        }
        frame = pd.DataFrame([row], columns=IMPUTATION_FEATURES)
        try:
            raw = float(self.metric_models[name].predict(frame)[0])
            value = max(math.expm1(raw), 0.0)
            if not finite(value) or value > (100 if name == "duration_mod" else 10000):
                raise ValueError("implausible model output")
            response.value = value
            response.source = pb.METRIC_SOURCE_CATBOOST
            response.model_version = str(deployment["model_version"])
            response.review_required = True
            response.limitations.extend(deployment.get("limitations", []))
            interval_model = self.interval_models.get(name)
            regime = "available_features" if features.HasField("ytm") else "without_yield"
            radius = self.metric_metadata.get("calibration_radii_log", {}).get(f"{name}:{regime}")
            if interval_model is not None and radius is not None:
                quantiles = np.sort(np.asarray(interval_model.predict(frame)).reshape(-1))
                if len(quantiles) != 3:
                    raise ValueError("invalid quantile output")
                low = max(math.expm1(float(quantiles[0]) - float(radius)), 0.0)
                high = max(math.expm1(float(quantiles[2]) + float(radius)), value)
                if not finite(low) or not finite(high):
                    raise ValueError("invalid interval")
                response.low = min(low, value)
                response.high = high
            else:
                response.limitations.append("calibrated interval unavailable")
            return response
        except (ValueError, OverflowError, TypeError, CatBoostError) as exc:
            raise ModelError("UNAVAILABLE", "model inference failed") from exc

    def analyze(self, request: pb.AnalyzeScenarioRequest) -> pb.AnalyzeScenarioResponse:
        if not self.scenario_ready:
            raise ModelError("FAILED_PRECONDITION", self.scenario_error)
        metadata = self.scenario_metadata
        if not finite(request.shift_bp) or not float(metadata["shift_bp_min"]) <= request.shift_bp <= float(metadata["shift_bp_max"]):
            raise ModelError("OUT_OF_RANGE", "shift_bp is outside the validated range")
        if not request.positions or len(request.positions) > 1000:
            raise ModelError("INVALID_ARGUMENT", "positions must contain 1 to 1000 items")
        rows = []
        seen: set[str] = set()
        for position in request.positions:
            if not position.position_id or position.position_id in seen:
                raise ModelError("INVALID_ARGUMENT", "duplicate or missing position id")
            seen.add(position.position_id)
            for field in ("duration_mod", "ytm", "face_value"):
                if not position.HasField(field) or not finite(getattr(position, field)):
                    raise ModelError("INVALID_ARGUMENT", f"{field} is required")
            if position.duration_mod < 0 or position.face_value <= 0 or position.ytm <= -100:
                raise ModelError("INVALID_ARGUMENT", "invalid duration, face value or yield")
            for name, bounds in metadata.get("input_ranges", {}).items():
                if not float(bounds[0]) <= getattr(position, name) <= float(bounds[1]):
                    raise ModelError("OUT_OF_RANGE", f"{name} is outside the training range")
            rows.append({"shock_abs": request.shift_bp / 10000, "DurationMod": position.duration_mod,
                         "YIELDCLOSE": position.ytm, "FACEVALUE": position.face_value})
        try:
            # The target formula is identically zero at dy=0. Do not expose a
            # tree model's numerical residual as an effect of a zero shock.
            predictions = (np.zeros(len(rows)) if request.shift_bp == 0 else
                           np.asarray(self.scenario_model.predict(pd.DataFrame(rows, columns=SCENARIO_FEATURES))).reshape(-1))
            if len(predictions) != len(rows) or not np.isfinite(predictions).all():
                raise ValueError("invalid scenario output")
        except (ValueError, OverflowError, TypeError, CatBoostError) as exc:
            raise ModelError("UNAVAILABLE", "scenario inference failed") from exc
        response = pb.AnalyzeScenarioResponse(
            request_id=request.request_id, calculation_id=request.calculation_id,
            model_version=str(metadata["model_version"]), target_kind=str(metadata["target_kind"]),
            target_unit=str(metadata["target_unit"]), experimental=bool(metadata.get("experimental", False)),
        )
        response.limitations.extend(metadata.get("limitations", []))
        tested_shifts = metadata.get("tested_shifts_bp")
        if tested_shifts is not None and request.shift_bp not in tested_shifts:
            response.limitations.append("This intermediate shift was not separately evaluated; model interpolation is approximate.")
        for position, value in zip(request.positions, predictions, strict=True):
            response.positions.add(position_id=position.position_id, model_value=float(value))
        return response
