from unittest.mock import Mock, patch

from src.rag_backend.models.config import Config


def test_config_loading():
    config_data = {
        "venice_api_base": "https://api.venice.ai/api/v1",
        "models": {
            "test-model": {
                "name": "gpt-3.5-turbo",
                "temperature": 0.7
            }
        },
        "configurations": {
            "default": {
                "data_directory": "data",
                "vector_db_config": {
                    "collection_name": "test_collection",
                    "embedding_model": "text-embedding-3-small",
                    "chunk_size": 500,
                    "chunk_overlap": 50,
                    "top_k": 5,
                },
                "pipeline_config": {
                    "main_prompt": {
                        "system_prompt": "Test system prompt",
                        "user_prompt_template": "Test {question} with {context}",
                        "model": "test-model",
                    },
                    "gate_prompts": [],
                    "rewrite_prompts": [],
                    "max_retries": 2,
                }
            }
        }
    }

    config = Config(**config_data)
    assert config.venice_api_base == "https://api.venice.ai/api/v1"
    assert "default" in config.configurations
    assert config.configurations["default"].data_directory == "data"
    assert config.configurations["default"].vector_db_config.collection_name == "test_collection"
    assert config.configurations["default"].pipeline_config.main_prompt.system_prompt == "Test system prompt"


def test_multi_config_loading():
    config_data = {
        "venice_api_base": "https://api.venice.ai/api/v1",
        "models": {
            "test-model": {
                "name": "gpt-3.5-turbo",
                "temperature": 0.7
            }
        },
        "configurations": {
            "default": {
                "data_directory": "data",
                "vector_db_config": {
                    "collection_name": "test_collection",
                    "embedding_model": "text-embedding-3-small",
                    "chunk_size": 500,
                    "chunk_overlap": 50,
                    "top_k": 5,
                },
                "pipeline_config": {
                    "main_prompt": {
                        "system_prompt": "Test system prompt",
                        "user_prompt_template": "Test {question} with {context}",
                        "model": "test-model",
                    },
                    "gate_prompts": [],
                    "rewrite_prompts": [],
                    "max_retries": 2,
                }
            },
            "liberation": {
                "data_directory": "data-liberation",
                "vector_db_config": {
                    "collection_name": "liberation_collection",
                    "embedding_model": "text-embedding-3-small",
                    "chunk_size": 300,
                    "chunk_overlap": 30,
                    "top_k": 3,
                },
                "pipeline_config": {
                    "main_prompt": {
                        "system_prompt": "Liberation system prompt",
                        "user_prompt_template": "Liberation {question} with {context}",
                        "model": "test-model",
                    },
                    "gate_prompts": [],
                    "rewrite_prompts": [],
                    "max_retries": 1,
                }
            }
        }
    }

    config = Config(**config_data)
    assert config.venice_api_base == "https://api.venice.ai/api/v1"
    assert len(config.configurations) == 2
    assert "default" in config.configurations
    assert "liberation" in config.configurations
    assert config.configurations["default"].data_directory == "data"
    assert config.configurations["liberation"].data_directory == "data-liberation"
    assert config.configurations["default"].vector_db_config.collection_name == "test_collection"
    assert config.configurations["liberation"].vector_db_config.collection_name == "liberation_collection"


def test_api_endpoint():
    with patch.dict("os.environ", {"VENICE_API_KEY": "test-key"}):
        from src.rag_backend.app import create_app

        mock_vector_db = Mock()
        mock_vector_db.load_or_create_index = Mock()
        mock_vector_db.get_context = Mock(return_value="Test context")

        mock_pipeline = Mock()
        mock_pipeline.run_pipeline = Mock(return_value="Test response")

        mock_config_manager = Mock()
        mock_config_manager.has_configuration = Mock(return_value=True)
        mock_config_manager.get_vector_db_service = Mock(return_value=mock_vector_db)
        mock_config_manager.get_pipeline_service = Mock(return_value=mock_pipeline)
        mock_config_manager.get_available_models = Mock(return_value=["default"])

        with patch("src.rag_backend.app.ConfigurationManager", return_value=mock_config_manager):
            app = create_app("config.json", reindex=False)
            client = app.test_client()

            response = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Test question"}],
                    "stream": False,
                    "model": "default"
                },
            )

            assert response.status_code == 200
            data = response.get_json()
            assert data["object"] == "chat.completion"
            assert data["choices"][0]["message"]["content"] == "Test response"
