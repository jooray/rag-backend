import os
import json
import argparse
from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from .models.config import Config
from .services.config_manager import ConfigurationManager
from .api.completions import api_bp, init_configuration_manager


def create_app(config_path: str = "config.json", reindex: bool = False) -> Flask:
    load_dotenv()

    if not os.getenv("VENICE_API_KEY"):
        raise ValueError("VENICE_API_KEY environment variable is required")

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    config = Config(**config_dict)

    config_manager = ConfigurationManager(
        config=config, api_key=os.getenv("VENICE_API_KEY"), reindex=reindex
    )
    init_configuration_manager(config_manager)

    app = Flask(__name__)
    app.register_blueprint(api_bp)

    cors_config = config.server_config.cors
    CORS(
        app,
        origins=cors_config.origins,
        methods=cors_config.methods,
        allow_headers=cors_config.headers,
        supports_credentials=cors_config.supports_credentials,
    )

    @app.route("/health", methods=["GET"])
    def health():  # noqa: D401
        return jsonify({"status": "ok"})

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
    parser.add_argument(
        "--cors-origin",
        action="append",
        help="Add allowed CORS origin (can be specified multiple times)",
    )
    parser.add_argument(
        "--cors-method",
        action="append",
        help="Add allowed CORS HTTP method (can be specified multiple times)",
    )
    parser.add_argument(
        "--cors-header",
        action="append",
        help="Add allowed CORS header (can be specified multiple times)",
    )
    parser.add_argument(
        "--cors-credentials", action="store_true", help="Allow credentials in CORS requests"
    )
    parser.add_argument("--mmr", action="store_true", help="Enable MMR search")
    parser.add_argument("--no-mmr", action="store_true", help="Disable MMR search")
    parser.add_argument(
        "--mmr-lambda", type=float, help="MMR lambda parameter (0=diversity, 1=relevance)"
    )
    parser.add_argument(
        "--pipeline-max-workers", type=int, help="Override pipeline max worker threads"
    )

    args = parser.parse_args()

    load_dotenv()

    if not os.getenv("VENICE_API_KEY"):
        raise SystemExit("VENICE_API_KEY environment variable is required")

    with open(args.config, "r") as f:
        config_dict = json.load(f)

    config = Config(**config_dict)

    if args.host:
        config.server_config.host = args.host
    if args.port:
        config.server_config.port = args.port
    if args.debug:
        config.server_config.debug = True

    if args.cors_origin:
        config.server_config.cors.origins.extend(args.cors_origin)
    if args.cors_method:
        config.server_config.cors.methods.extend(args.cors_method)
    if args.cors_header:
        config.server_config.cors.headers.extend(args.cors_header)
    if args.cors_credentials:
        config.server_config.cors.supports_credentials = True

    if args.mmr:
        for entry in config.configurations.values():
            entry.vector_db_config.use_mmr = True
    if args.no_mmr:
        for entry in config.configurations.values():
            entry.vector_db_config.use_mmr = False
    if args.mmr_lambda is not None:
        for entry in config.configurations.values():
            entry.vector_db_config.mmr_lambda = args.mmr_lambda

    if args.pipeline_max_workers is not None:
        config.server_config.pipeline_max_workers = args.pipeline_max_workers

    config_manager = ConfigurationManager(
        config=config, api_key=os.getenv("VENICE_API_KEY"), reindex=args.reindex
    )
    init_configuration_manager(
        config_manager, max_workers=config.server_config.pipeline_max_workers
    )

    app = Flask(__name__)
    app.register_blueprint(api_bp)

    cors_config = config.server_config.cors
    CORS(
        app,
        origins=cors_config.origins,
        methods=cors_config.methods,
        allow_headers=cors_config.headers,
        supports_credentials=cors_config.supports_credentials,
    )

    @app.route("/health", methods=["GET"])
    def _health():
        return jsonify({"status": "ok"})

    app.run(
        host=config.server_config.host,
        port=config.server_config.port,
        debug=config.server_config.debug,
        threaded=True,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
