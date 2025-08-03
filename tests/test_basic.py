from unittest.mock import Mock, patch

from src.rag_backend.models.config import Config


def test_config_loading():
    config_data = {
        "venice_api_base": "https://api.venice.ai/api/v1",
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
                "model_config": {"name": "gpt-3.5-turbo", "temperature": 0.7},
            },
            "gate_prompts": [],
            "rewrite_prompts": [],
            "max_retries": 2,
        },
    }

    config = Config(**config_data)
    assert config.venice_api_base == "https://api.venice.ai/api/v1"
    assert config.data_directory == "data"
    assert config.vector_db_config.collection_name == "test_collection"
    assert config.pipeline_config.main_prompt.system_prompt == "Test system prompt"


def test_api_endpoint():
    with patch.dict("os.environ", {"VENICE_API_KEY": "test-key"}):
        from src.rag_backend.app import create_app

        mock_vector_db = Mock()
        mock_vector_db.load_or_create_index = Mock()
        mock_vector_db.get_context = Mock(return_value="Test context")

        mock_pipeline = Mock()
        mock_pipeline.run_pipeline = Mock(return_value="Test response")

        with patch("src.rag_backend.app.VectorDBService", return_value=mock_vector_db):
            with patch(
                "src.rag_backend.app.PipelineService", return_value=mock_pipeline
            ):
                app = create_app("config.json", reindex=False)
                client = app.test_client()

                response = client.post(
                    "/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": "Test question"}],
                        "stream": False,
                    },
                )

                assert response.status_code == 200
                data = response.get_json()
                assert data["object"] == "chat.completion"
                assert data["choices"][0]["message"]["content"] == "Test response"
