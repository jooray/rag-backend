# RAG Backend

An OpenAI-compatible API server with Retrieval-Augmented Generation (RAG) capabilities and a sophisticated filtering pipeline.

## Prerequisites

- **Python 3.8+** with Poetry for dependency management
- **Ollama** running locally for embeddings (default: http://localhost:11434)
  - Install from: https://ollama.ai/
  - Pull the embedding model: `ollama pull nomic-embed-text`
- **Venice API key** for LLM responses

## Quick Start

```bash
# 1. Install and start Ollama
# Download from https://ollama.ai/ and start the service
ollama pull nomic-embed-text

# 2. Install dependencies
poetry install

# 3. Set up environment
cp .env.example .env
# Edit .env and add your Venice API key

# 4. Add data files to data/ directory
# - .txt files for documents
# - .jsonl files for Q&A pairs

# 5. Run the server
poetry run python -m src.rag_backend.app

# 6. Test the API
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is AI?"}]}'
```

## Features

- **OpenAI-compatible API**: Implements `/v1/chat/completions` endpoint
- **RAG Support**: Index text files and JSONL Q&A pairs for context-aware responses
- **Local Embeddings**: Uses Ollama for local embedding generation (no external API calls)
- **Filtering Pipeline**: Gate prompts for quality control with automatic fix attempts
- **Rewrite Pipeline**: Transform responses through multiple rewrite stages
- **Flexible Configuration**: JSON-based configuration for all prompts and models
- **Venice.ai Integration**: Uses Venice.ai as the LLM backend

## Installation

```bash
poetry install
```

## Configuration

1. Set the `VENICE_API_KEY` environment variable:
```bash
export VENICE_API_KEY="your-venice-api-key"
```

2. Ensure Ollama is running with the embedding model:
```bash
ollama serve  # Start Ollama service
ollama pull nomic-embed-text  # Pull the default embedding model
```

3. Create a `config.json` file (see example provided)

### Embedding Configuration

The vector database uses Ollama for local embeddings. You can configure this in `config.json`:

```json
{
  "vector_db_config": {
    "embedding_model": "nomic-embed-text",
    "ollama_base_url": "http://localhost:11434",
    "collection_name": "rag_documents",
    "chunk_size": 500,
    "chunk_overlap": 50,
    "top_k": 5
  }
}
```

- `embedding_model`: Ollama model to use for embeddings (default: "nomic-embed-text")
- `ollama_base_url`: Ollama API endpoint (default: "http://localhost:11434")
- Other vector database settings can also be configured

### Data Files

4. Add your data files to the `data/` directory:
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
  --host HOST      Host to bind to (default: 0.0.0.0)
  --port PORT      Port to bind to (default: 8000)
  --debug          Run in debug mode
```

### API Usage

Send requests to the OpenAI-compatible endpoint:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Your question here"}],
    "stream": false
  }'
```

## Pipeline Architecture

1. **Main Prompt**: Generates initial response with RAG context
2. **Gate Prompts**: Check response quality (factuality, safety, etc.)
   - If REJECT: Run fix prompt and retry (up to max_retries)
3. **Rewrite Prompts**: Polish the final response (conciseness, tone, etc.)

## Configuration Guide

### Vector Database Config
- `embedding_model`: Model for creating embeddings
- `chunk_size`: Size of text chunks for indexing
- `chunk_overlap`: Overlap between chunks
- `top_k`: Number of similar documents to retrieve

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
