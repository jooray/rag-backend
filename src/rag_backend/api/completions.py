from typing import Dict, Any, Optional
from flask import Blueprint, request, jsonify, Response

import json
import re
import time
import uuid

from ..services.config_manager import ConfigurationManager


api_bp = Blueprint("api", __name__)

configuration_manager: Optional[ConfigurationManager] = None


def init_configuration_manager(config_manager: ConfigurationManager):
    global configuration_manager
    configuration_manager = config_manager


@api_bp.route("/v1/chat/completions", methods=["POST", "OPTIONS"])
def chat_completions():
    # Handle OPTIONS preflight request
    if request.method == "OPTIONS":
        return "", 200

    try:
        data = request.json

        messages = data.get("messages", [])
        stream = data.get("stream", False)
        model = data.get("model", "default")  # Default to "default" configuration

        if not messages:
            return (
                jsonify(
                    {
                        "error": {
                            "message": "Messages array is required",
                            "type": "invalid_request_error",
                            "code": "invalid_request",
                        }
                    }
                ),
                400,
            )

        # Check if the requested model configuration exists
        if not configuration_manager.has_configuration(model):
            available_models = configuration_manager.get_available_models()
            return (
                jsonify(
                    {
                        "error": {
                            "message": f"Model '{model}' not found. Available models: {', '.join(available_models)}",
                            "type": "invalid_request_error",
                            "code": "model_not_found",
                        }
                    }
                ),
                400,
            )

        last_message = messages[-1]
        if last_message.get("role") != "user":
            return (
                jsonify(
                    {
                        "error": {
                            "message": "Last message must be from user",
                            "type": "invalid_request_error",
                            "code": "invalid_request",
                        }
                    }
                ),
                400,
            )

        question = last_message.get("content", "")

        # Get the appropriate services for the selected model
        vector_db_service = configuration_manager.get_vector_db_service(model)
        pipeline_service = configuration_manager.get_pipeline_service(model)

        context = vector_db_service.get_context(question)
        response_text = pipeline_service.run_pipeline(messages, context)

        if stream:
            return create_stream_response(response_text, data)
        else:
            return create_completion_response(response_text, data)

    except Exception as e:
        return (
            jsonify(
                {
                    "error": {
                        "message": str(e),
                        "type": "internal_server_error",
                        "code": "internal_error",
                    }
                }
            ),
            500,
        )


def create_completion_response(
    content: str, original_request: Dict[str, Any]
) -> Dict[str, Any]:
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


def create_stream_response(content: str, original_request: Dict[str, Any]) -> Response:
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    model = original_request.get("model", "rag-backend")

    def generate():
        # Split content while preserving whitespace by using regex
        # Split on word boundaries but capture the separating whitespace
        tokens = re.split(r'(\s+)', content)
        # Filter out empty strings
        tokens = [token for token in tokens if token]

        for token in tokens:
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": token
                        },
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        final_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@api_bp.route("/v1/models", methods=["GET", "OPTIONS"])
def list_models():
    # Handle OPTIONS preflight request
    if request.method == "OPTIONS":
        return "", 200

    available_models = configuration_manager.get_available_models()
    model_data = []

    for model_name in available_models:
        model_data.append({
            "id": model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "rag-backend",
        })

    return jsonify(
        {
            "object": "list",
            "data": model_data,
        }
    )
