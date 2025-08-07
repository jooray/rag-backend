from typing import Dict, Any, Optional, List, Generator
import os
import uuid
import time
import json
import re
from concurrent.futures import ThreadPoolExecutor

from flask import Blueprint, request, jsonify, Response

from ..services.config_manager import ConfigurationManager


api_bp = Blueprint("api", __name__)

configuration_manager: Optional[ConfigurationManager] = None
_executor: Optional[ThreadPoolExecutor] = None


def init_configuration_manager(config_manager: ConfigurationManager, max_workers: int | None = None):
    global configuration_manager, _executor
    configuration_manager = config_manager
    # Initialize executor lazily with configured workers (fallback to env for backwards compatibility)
    if _executor is None:
        if max_workers is None and configuration_manager is not None:
            max_workers = configuration_manager.config.server_config.pipeline_max_workers
        if max_workers is None:  # final fallback
            max_workers = int(os.getenv("PIPELINE_MAX_WORKERS", "4"))
        _executor = ThreadPoolExecutor(max_workers=max_workers)


def _error(message: str, status: int):
    return jsonify({"error": {"message": message, "type": "invalid_request_error"}}), status


@api_bp.route("/v1/chat/completions", methods=["POST", "OPTIONS"])
def chat_completions():
    if request.method == "OPTIONS":
        # CORS preflight; Flask-CORS will fill headers
        return ("", 204)

    if configuration_manager is None or _executor is None:
        return _error("Server not initialized", 500)

    try:
        data = request.get_json(silent=True) or {}
        messages: List[Dict[str, Any]] = data.get("messages", [])
        stream: bool = bool(data.get("stream", False))
        model: str = data.get("model", "default")

        if not messages:
            return _error("'messages' is required and must be non-empty", 400)

        if not configuration_manager.has_configuration(model):
            return _error(f"Unknown model configuration '{model}'", 400)

        last_message = messages[-1]
        if last_message.get("role") != "user":
            return _error("Last message must have role 'user'", 400)

        vector_db_service = configuration_manager.get_vector_db_service(model)
        pipeline_service = configuration_manager.get_pipeline_service(model)
        if vector_db_service is None or pipeline_service is None:
            return _error("Configuration services not available", 500)

        # Vector search (can be blocking; acceptable per requirements)
        context = vector_db_service.get_context(last_message.get("content", ""))

        # Run pipeline concurrently so multiple requests can progress in parallel
        future = _executor.submit(pipeline_service.run_pipeline, messages, context)

        if stream:
            # Build streaming response once future completes; we still offload heavy work
            def generate() -> Generator[str, None, None]:
                try:
                    content = future.result()
                    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
                    created = int(time.time())
                    model_name = model

                    tokens = re.split(r"(\s+)", content)
                    tokens = [t for t in tokens if t]

                    for token in tokens:
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": token},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

                    final_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {"index": 0, "delta": {}, "finish_reason": "stop"}
                        ],
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:  # noqa: BLE001
                    err_chunk = {"error": str(e)}
                    yield f"data: {json.dumps(err_chunk)}\n\n"
                    yield "data: [DONE]\n\n"

            return Response(
                generate(),
                mimetype="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        else:
            content = future.result()
            return create_completion_response(content, data)

    except Exception as e:  # noqa: BLE001
        return _error(str(e), 500)


def create_completion_response(
    content: str, original_request: Dict[str, Any]
) -> Response:
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    return jsonify(
        {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": original_request.get("model", "rag-backend"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1},
        }
    )


@api_bp.route("/v1/models", methods=["GET", "OPTIONS"])
def list_models():
    if request.method == "OPTIONS":
        return ("", 204)

    if configuration_manager is None:
        return _error("Server not initialized", 500)

    available_models = configuration_manager.get_available_models()
    model_data = []

    for model_name in available_models:
        model_data.append(
            {
                "id": model_name,
                "object": "model",
                "created": 0,
                "owned_by": "rag-backend",
            }
        )

    return jsonify(
        {
            "object": "list",
            "data": model_data,
        }
    )
