"""Versioned Python gRPC server. Run with: python -m ml_service.server."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import Callable
from concurrent import futures
from datetime import date
from pathlib import Path
from typing import Any

import grpc
from google.protobuf.any_pb2 import Any as AnyMessage
from google.rpc import error_details_pb2, status_pb2
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from grpc_status import rpc_status

from .logic import Provider, ServiceError, explain_calculation, provider_from_env
from .models import ModelError, ModelManager
from .v1 import service_pb2 as pb
from .v1 import service_pb2_grpc as pb_grpc

logger = logging.getLogger(__name__)
_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event = {"level": record.levelname, "message": record.getMessage()}
        for name in ("rpc_method", "request_id", "status", "elapsed_ms", "error_type", "metric_ready", "scenario_ready", "explanation_ready"):
            if hasattr(record, name):
                event[name] = getattr(record, name)
        return json.dumps(event, ensure_ascii=False)


def _identity(request_id: str, subject_id: str, as_of_date: str) -> None:
    if not _ID.fullmatch(request_id) or not _ID.fullmatch(subject_id):
        raise ModelError("INVALID_ARGUMENT", "invalid request or subject id")
    try:
        date.fromisoformat(as_of_date)
    except ValueError as exc:
        raise ModelError("INVALID_ARGUMENT", "date must be YYYY-MM-DD") from exc


def _abort(context: grpc.ServicerContext, code: str, message: str, request_id: str, method: str) -> None:
    status_code = getattr(grpc.StatusCode, code, grpc.StatusCode.INTERNAL)
    detail = error_details_pb2.ErrorInfo(
        reason=code, domain="ml_service.v1",
        metadata={"request_id": request_id, "method": method},
    )
    packed = AnyMessage()
    packed.Pack(detail)
    status = status_pb2.Status(code=status_code.value[0], message=message, details=[packed])
    context.abort_with_status(rpc_status.to_status(status))


class MLService(pb_grpc.MLServiceServicer):
    def __init__(self, models: ModelManager, provider: Provider | None):
        self.models = models
        self.provider = provider

    def _run(self, context: grpc.ServicerContext, request_id: str, method: str, action: Callable[[], Any]) -> Any:
        start = time.monotonic()
        log_id = request_id if _ID.fullmatch(request_id) else "invalid"
        try:
            result = action()
            logger.info("rpc completed", extra={
                "rpc_method": method, "request_id": log_id, "status": "OK",
                "elapsed_ms": int((time.monotonic() - start) * 1000),
            })
            return result
        except (ModelError, ServiceError) as exc:
            logger.warning("rpc rejected", extra={
                "rpc_method": method, "request_id": log_id, "status": exc.code,
                "elapsed_ms": int((time.monotonic() - start) * 1000),
            })
            _abort(context, exc.code, str(exc), log_id, method)
        except Exception as exc:  # noqa: BLE001 - gRPC must turn unexpected failures into INTERNAL.
            logger.error("rpc failed", extra={
                "rpc_method": method, "request_id": log_id, "status": "INTERNAL",
                "error_type": type(exc).__name__,
            })
            _abort(context, "INTERNAL", "internal ML service error", log_id, method)

    def EstimateMissingMetric(self, request: pb.EstimateMissingMetricRequest, context: grpc.ServicerContext) -> pb.EstimateMissingMetricResponse:
        def action() -> pb.EstimateMissingMetricResponse:
            _identity(request.request_id, request.instrument_id, request.as_of_date)
            return self.models.estimate(request)

        return self._run(context, request.request_id, "EstimateMissingMetric", action)

    def AnalyzeScenario(self, request: pb.AnalyzeScenarioRequest, context: grpc.ServicerContext) -> pb.AnalyzeScenarioResponse:
        def action() -> pb.AnalyzeScenarioResponse:
            _identity(request.request_id, request.calculation_id, request.data_date)
            return self.models.analyze(request)

        return self._run(context, request.request_id, "AnalyzeScenario", action)

    def ExplainCalculation(self, request: pb.ExplainCalculationRequest, context: grpc.ServicerContext) -> pb.ExplainCalculationResponse:
        def action() -> pb.ExplainCalculationResponse:
            _identity(request.request_id, request.calculation_id, request.data_as_of)
            kinds = {pb.CALCULATION_KIND_DV01: "DV01", pb.CALCULATION_KIND_STRESS_CHANGE: "STRESS_CHANGE"}
            if request.calculation_kind not in kinds or len(request.positions) > 1000:
                raise ModelError("INVALID_ARGUMENT", "invalid calculation kind or positions")
            positions = []
            for position in request.positions:
                if position.side not in ("buy", "sell"):
                    raise ModelError("INVALID_ARGUMENT", "side must be buy or sell")
                entry = {
                    "id": position.position_id, "side": position.side.upper(),
                    "value_rub": position.value_rub, "change_rub": position.change_rub,
                    "estimated": position.estimated,
                }
                if position.isin:
                    entry["isin"] = position.isin
                if position.HasField("inputs"):
                    inputs = position.inputs
                    sources = {
                        pb.METRIC_SOURCE_OBSERVED: "observed",
                        pb.METRIC_SOURCE_CATBOOST: "catboost",
                        pb.METRIC_SOURCE_COUPON_RULE: "coupon_rule",
                    }
                    entry["inputs"] = {
                        "current_fair_value": inputs.current_fair_value,
                        "position_currency": inputs.position_currency,
                        "fx_rate_to_rub": inputs.fx_rate_to_rub,
                        "duration_source": sources.get(inputs.duration_source, "unknown"),
                        "convexity_source": sources.get(inputs.convexity_source, "unknown"),
                        "coupon_type": inputs.coupon_type,
                    }
                    for name in ("duration_mod", "convexity", "coupon_frequency"):
                        if inputs.HasField(name):
                            entry["inputs"][name] = getattr(inputs, name)
                positions.append(entry)
            data = {
                "calculation_id": request.calculation_id,
                "calculation_kind": kinds[request.calculation_kind],
                "currency": request.currency, "total_rub": request.total_rub,
                "data_as_of": request.data_as_of, "incomplete": request.incomplete,
                "estimated": request.estimated, "positions": positions,
            }
            if request.method == pb.EXPLANATION_METHOD_DURATION_CONVEXITY:
                data["method"] = "duration_convexity"
            elif request.method != pb.EXPLANATION_METHOD_UNSPECIFIED:
                raise ModelError("INVALID_ARGUMENT", "unknown explanation method")
            if request.HasField("yield_shift_bp"):
                data["yield_shift_bp"] = request.yield_shift_bp
            result = explain_calculation(data, self.provider, timeout=context.time_remaining())
            return pb.ExplainCalculationResponse(
                request_id=request.request_id, calculation_id=result["calculation_id"],
                text=result["text"], fact_ids=result["fact_ids"],
                model_version=result["model_version"], verified=result["verified"],
                calculation_details_available=result["calculation_details_available"],
                limitations=result["limitations"],
            )

        return self._run(context, request.request_id, "ExplainCalculation", action)

    def GetReadiness(self, _request: pb.ReadinessRequest, context: grpc.ServicerContext) -> pb.ReadinessResponse:
        def action() -> pb.ReadinessResponse:
            result = pb.ReadinessResponse(
                metric_ready=self.models.metric_ready,
                scenario_ready=self.models.scenario_ready,
                explanation_ready=self.provider is not None,
            )
            if not result.metric_ready:
                result.reasons.append(self.models.metric_error)
            if not result.scenario_ready:
                result.reasons.append(self.models.scenario_error)
            if not result.explanation_ready:
                result.reasons.append("LLM provider not configured")
            return result

        return self._run(context, "health", "GetReadiness", action)


def create_server(models: ModelManager, provider: Provider | None, address: str) -> tuple[grpc.Server, int]:
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=8),
        options=[("grpc.max_receive_message_length", 1024 * 1024), ("grpc.max_send_message_length", 1024 * 1024)],
    )
    pb_grpc.add_MLServiceServicer_to_server(MLService(models, provider), server)
    health_service = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_service, server)
    health_service.set("", health_pb2.HealthCheckResponse.SERVING)
    health_service.set("ml_service.v1.MLService", health_pb2.HealthCheckResponse.SERVING)
    port = server.add_insecure_port(address)
    if not port:
        raise RuntimeError("cannot bind ML gRPC service")
    return server, port


def serve() -> None:
    models = ModelManager(Path(os.getenv("MODEL_DIR", "./models")))
    try:
        provider = provider_from_env()
    except ServiceError:
        provider = None
    address = os.getenv("ML_GRPC_LISTEN", "127.0.0.1:50051")
    server, _ = create_server(models, provider, address)
    server.start()
    logger.info("ML gRPC server started", extra={
        "metric_ready": models.metric_ready, "scenario_ready": models.scenario_ready,
        "explanation_ready": provider is not None,
    })
    server.wait_for_termination()


if __name__ == "__main__":
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    serve()
