from typing import Dict, Any, Optional
from flask import Blueprint, request, jsonify, Response

import json
import time
import uuid

from ..services.vector_db import VectorDBService
from ..services.pipeline import PipelineService


api_bp = Blueprint("api", __name__)

vector_db_service: Optional[VectorDBService] = None
pipeline_service: Optional[PipelineService] = None


def init_services(vector_db: VectorDBService, pipeline: PipelineService):
    global vector_db_service, pipeline_service
    vector_db_service = vector_db
    pipeline_service = pipeline


@api_bp.route("/v1/chat/completions", methods=["POST", "OPTIONS"])
def chat_completions():
    # Handle OPTIONS preflight request
    if request.method == "OPTIONS":
        return "", 200

    try:
        data = request.json

        messages = data.get("messages", [])
        stream = data.get("stream", False)

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

        context = vector_db_service.get_context(question)

        response_text = pipeline_service.run_pipeline(question, context)

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
        words = content.split()
        for i, word in enumerate(words):
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": word + (" " if i < len(words) - 1 else "")
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

    return jsonify(
        {
            "object": "list",
            "data": [
                {
                    "id": "rag-backend",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "rag-backend",
                }
            ],
        }
    )
