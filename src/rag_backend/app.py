import os
import json
import argparse

from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv

from .models.config import Config
from .services.vector_db import VectorDBService
from .services.pipeline import PipelineService
from .api.completions import api_bp, init_services


def create_app(config_path: str = "config.json", reindex: bool = False) -> Flask:
    load_dotenv()

    if not os.getenv("VENICE_API_KEY"):
        raise ValueError("VENICE_API_KEY environment variable is required")

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    config = Config(**config_dict)

    vector_db = VectorDBService(
        config=config.vector_db_config,
        data_dir=config.data_directory,
    )
    vector_db.load_or_create_index(reindex=reindex)

    pipeline = PipelineService(
        config=config.pipeline_config,
        models=config.models,
        api_key=os.getenv("VENICE_API_KEY")
    )

    init_services(vector_db, pipeline)

    app = Flask(__name__)
    app.register_blueprint(api_bp)

    # Configure CORS
    cors_config = config.server_config.cors
    CORS(
        app,
        origins=cors_config.origins,
        methods=cors_config.methods,
        allow_headers=cors_config.headers,
        supports_credentials=cors_config.supports_credentials
    )

    @app.route("/health", methods=["GET"])
    def health():
        return {"status": "ok"}

    return app


def main():
    parser = argparse.ArgumentParser(description="RAG Backend Server")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    parser.add_argument(
        "--reindex", action="store_true", help="Force reindex of documents"
    )
    parser.add_argument("--host", help="Host to bind to (overrides config)")
    parser.add_argument("--port", type=int, help="Port to bind to (overrides config)")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--cors-origin", action="append", help="Add allowed CORS origin (can be specified multiple times)")
    parser.add_argument("--cors-method", action="append", help="Add allowed CORS HTTP method (can be specified multiple times)")
    parser.add_argument("--cors-header", action="append", help="Add allowed CORS header (can be specified multiple times)")
    parser.add_argument("--cors-credentials", action="store_true", help="Allow credentials in CORS requests")

    args = parser.parse_args()

    # Load config file
    with open(args.config, "r") as f:
        config_dict = json.load(f)
    config = Config(**config_dict)

    # Override config with CLI args
    if args.host:
        config.server_config.host = args.host
    if args.port:
        config.server_config.port = args.port
    if args.debug:
        config.server_config.debug = args.debug

    # Handle CORS CLI overrides
    if args.cors_origin:
        config.server_config.cors.origins = args.cors_origin
    if args.cors_method:
        config.server_config.cors.methods = args.cors_method
    if args.cors_header:
        config.server_config.cors.headers = args.cors_header
    if args.cors_credentials:
        config.server_config.cors.supports_credentials = True

    vector_db = VectorDBService(
        config=config.vector_db_config,
        data_dir=config.data_directory,
    )
    vector_db.load_or_create_index(reindex=args.reindex)

    pipeline = PipelineService(
        config=config.pipeline_config,
        models=config.models,
        api_key=os.getenv("VENICE_API_KEY")
    )

    init_services(vector_db, pipeline)

    app = Flask(__name__)
    app.register_blueprint(api_bp)

    # Configure CORS with updated settings
    cors_config = config.server_config.cors
    CORS(
        app,
        origins=cors_config.origins,
        methods=cors_config.methods,
        allow_headers=cors_config.headers,
        supports_credentials=cors_config.supports_credentials
    )

    # Add health check endpoint
    @app.route("/health", methods=["GET"])
    def health():
        return {"status": "ok"}

    app.run(
        host=config.server_config.host,
        port=config.server_config.port,
        debug=config.server_config.debug
    )


if __name__ == "__main__":
    main()
