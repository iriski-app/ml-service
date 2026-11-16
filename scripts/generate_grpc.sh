#!/usr/bin/env sh
set -eu

# Requires grpcio-tools==1.85.1. The Go backend generates its own client.
python -m grpc_tools.protoc -Iproto \
  --python_out=. --grpc_python_out=. \
  proto/ml_service/v1/service.proto
