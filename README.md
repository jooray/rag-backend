# RAG Backend

An OpenAI-compatible API server with Retrieval-Augmented Generation (RAG) capabilities, multi-configuration support, and a sophisticated filtering pipeline.

## Features

- **Multi-Configuration Support**: Define different data sources, pipelines, and settings for different models/use cases
- **OpenAI-compatible API**: Implements `/v1/chat/completions` endpoint
- **RAG Support**: Index text files and JSONL Q&A pairs for context-aware responses
- **Local Embeddings**: Uses Ollama for local embedding generation (no external API calls)
- **Filtering Pipeline**: Gate prompts for quality control with automatic fix attempts
- **Rewrite Pipeline**: Transform responses through multiple rewrite stages
- **Flexible Configuration**: JSON-based configuration for all prompts and models
- **Venice.ai Integration**: Uses Venice.ai as the LLM backend

## Prerequisites

- **Python 3.8+** with Poetry for dependency management
- **Ollama** running locally for embeddings (default: http://localhost:11434)
  - Install from: https://ollama.ai/
  - Pull the embedding model: `ollama pull mxbai-embed-large`
- **Venice API key** for LLM responses

## Quick Start

```bash
# 1. Install and start Ollama
# Download from https://ollama.ai/ and start the service
ollama pull mxbai-embed-large

# 2. Install dependencies
poetry install

# 3. Set up environment
cp .env.example .env
# Edit .env and add your Venice API key

# 4. Add data files to data/ directory (for default configuration)
# - .txt files for documents
# - .jsonl files for Q&A pairs

# 5. Run the server
poetry run python -m src.rag_backend.app

# 6. Test the API with default configuration
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "What is AI?"}]
  }'
```

## Installation

```bash
poetry install
```

## Multi-Configuration Support

The RAG backend supports multiple configurations, allowing you to define different data sources, pipelines, and settings for different models or use cases.

### Configuration Structure

#### Global Settings

These settings remain global across all configurations:

- `venice_api_base`: Venice API endpoint
- `server_config`: Server configuration (host, port, CORS, etc.)
- `models`: Model definitions used by all configurations

#### Individual Configurations

Each configuration is identified by a model name and can have its own:

- `data_directory`: Directory containing data files for this configuration
- `vector_db_config`: Vector database settings including collection name, embedding model, chunk settings
- `pipeline_config`: Pipeline configuration with prompts, gates, and rewrite rules

### Example Configuration

```json
{
  "venice_api_base": "https://api.venice.ai/api/v1",
  "server_config": {
    "host": "0.0.0.0",
    "port": 8080,
    "debug": false,
    "cors": {
      "origins": ["http://localhost:8000", "http://localhost:3000"],
      "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
      "headers": ["Content-Type", "Authorization", "X-Requested-With"],
      "supports_credentials": false
    }
  },
  "models": {
    "large": {
      "name": "qwen3-235b:strip_thinking_response=true",
      "temperature": 0.7,
      "max_tokens": 2000
    },
    "small": {
      "name": "venice-uncensored",
      "temperature": 0.7,
      "max_tokens": 2000
    }
  },
  "configurations": {
    "default": {
      "data_directory": "data",
      "vector_db_config": {
        "collection_name": "rag_documents",
        "embedding_model": "mxbai-embed-large",
        "ollama_base_url": "http://localhost:11434",
        "chunk_size": 500,
        "chunk_overlap": 50,
        "top_k": 5,
        "use_mmr": true,
        "mmr_fetch_k": 15,
        "mmr_lambda": 0.7
      },
      "pipeline_config": {
        "main_prompt": {
          "system_prompt": "You are a helpful assistant...",
          "user_prompt_template": "Context:\n{context}\n\nQuestion: {question}\n\nPlease provide a comprehensive answer.",
          "model": "large"
        },
        "gate_prompts": [...],
        "rewrite_prompts": [...],
        "max_retries": 2
      }
    },
    "liberation": {
      "data_directory": "data-liberation",
      "vector_db_config": {
        "collection_name": "liberation_documents",
        "embedding_model": "mxbai-embed-large",
        "ollama_base_url": "http://localhost:11434",
        "chunk_size": 300,
        "chunk_overlap": 30,
        "top_k": 3,
        "use_mmr": false
      },
      "pipeline_config": {
        "main_prompt": {
          "system_prompt": "You are a liberation-focused assistant...",
          "user_prompt_template": "Liberation Context:\n{context}\n\nQuestion: {question}\n\nProvide an informative answer focusing on liberation principles.",
          "model": "large"
        },
        "gate_prompts": [...],
        "rewrite_prompts": [...],
        "max_retries": 1
      }
    }
  }
}
```

## Configuration

1. Set the `VENICE_API_KEY` environment variable:
```bash
export VENICE_API_KEY="your-venice-api-key"
```

2. Ensure Ollama is running with the embedding model:
```bash
ollama serve  # Start Ollama service
ollama pull mxbai-embed-large  # Pull the default embedding model
```

3. Create a `config.json` file with your configurations (see example above)

### Setting Up Multiple Configurations

1. Create separate data directories for each configuration:
   ```bash
   mkdir data-liberation
   mkdir data-technical
   ```

2. Add your data files to each directory:
   ```bash
   cp liberation-documents/* data-liberation/
   cp technical-docs/* data-technical/
   ```

3. Configure different vector database collections for each configuration

4. Define appropriate prompts and pipelines for each use case

5. Start the server and use different model names in your API calls

### Data Files

Add your data files to the appropriate directories:
- `.txt` files: Plain text documents that will be chunked and indexed
- `.jsonl` files: Question-answer pairs in the format:
  ```json
  {"question": "...", "answer": "..."}
  ```

## Usage

### Start the server

```bash
poetry run python -m src.rag_backend.app
```

### Command-line options

```bash
poetry run python -m src.rag_backend.app --help

Options:
  --config PATH     Path to config file (default: config.json)
  --reindex        Force reindex of documents
  --host HOST      Host to bind to (overrides config)
  --port PORT      Port to bind to (overrides config)
  --debug          Run in debug mode
  --mmr            Enable MMR search for all configurations
  --no-mmr         Disable MMR search for all configurations
  --mmr-lambda     Set MMR lambda parameter for all configurations
```

### API Usage

#### Using Multiple Configurations

When making API calls, specify the configuration you want to use via the `model` parameter:

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "liberation",
    "messages": [
      {"role": "user", "content": "What is social justice?"}
    ]
  }'
```

#### Available Models

List available configurations:

```bash
curl http://localhost:8080/v1/models
```

This will return all configured model names that can be used in API calls.

#### Standard Usage

Send requests to the OpenAI-compatible endpoint:

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Your question here"}],
    "stream": false
  }'
```

## Pipeline Architecture

1. **Main Prompt**: Generates initial response with RAG context
2. **Gate Prompts**: Check response quality (factuality, safety, etc.)
   - If REJECT: Run fix prompt and retry (up to max_retries)
3. **Rewrite Prompts**: Polish the final response (conciseness, tone, etc.)

## Configuration Benefits

1. **Different Data Sources**: Each configuration can point to different data directories
2. **Specialized Pipelines**: Different prompts and processing rules for different use cases
3. **Separate Vector Stores**: Each configuration can have its own vector database collection
4. **Flexible Settings**: Different chunk sizes, retrieval parameters, and models per configuration
5. **Easy Switching**: Clients can switch between configurations by changing the model parameter

## Configuration Guide

### Vector Database Config
- `collection_name`: Name of the vector database collection
- `embedding_model`: Model for creating embeddings
- `ollama_base_url`: Ollama API endpoint
- `chunk_size`: Size of text chunks for indexing
- `chunk_overlap`: Overlap between chunks
- `top_k`: Number of similar documents to retrieve
- `use_mmr`: Enable Maximal Marginal Relevance for diverse results
- `mmr_fetch_k`: Number of documents to fetch before MMR reranking
- `mmr_lambda`: MMR lambda parameter (0=diversity, 1=relevance)

### Pipeline Config
- `main_prompt`: Initial response generation
- `gate_prompts`: Quality checks with optional fix prompts
- `rewrite_prompts`: Final transformations
- `max_retries`: Maximum fix attempts per gate

### Model Config
Each prompt can use a different model with its own temperature and token limits.

## Development

### Running tests
```bash
poetry run pytest tests/
```

### Code formatting
```bash
poetry run black .
```

### Linting
```bash
poetry run ruff check .
```

### Type checking
```bash
poetry run mypy src/
```
