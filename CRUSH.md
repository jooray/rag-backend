# rag-backend Development Guide

## Build/Test/Lint Commands
```bash
# Poetry environment
poetry install                                   # Install dependencies
poetry shell                                     # Activate virtual environment
poetry add package_name                          # Add dependency
poetry add --group dev package_name              # Add dev dependency

# Running the server
poetry run python -m src.rag_backend.app         # Start server
poetry run python -m src.rag_backend.app --debug # Debug mode
poetry run python -m src.rag_backend.app --reindex # Force reindex documents

# Testing
poetry run pytest tests/                         # Run all tests
poetry run pytest tests/test_specific.py         # Run single test file
poetry run pytest -k "test_function_name"        # Run specific test
poetry run pytest --cov=src                      # Run tests with coverage

# Code quality
poetry run black .                               # Format code
poetry run ruff check .                          # Lint code
poetry run mypy src/                             # Type check
```

## Code Style Guidelines
- **Imports**: Group stdlib, third-party (Flask/LangChain), local with blank lines between
- **Naming**: snake_case for variables/functions, PascalCase for classes, UPPER_CASE for constants
- **Types**: Always use type hints, import from typing module
- **Error Handling**: Use Flask error handlers, return proper HTTP status codes
- **Functions**: Keep functions small (<30 lines), single responsibility
- **Comments**: Minimal comments, code should be self-documenting
- **Flask**: Use blueprints for organization, dependency injection for services
- **LangChain**: Use langchain_openai, langchain_community, langchain_core packages
- **Environment**: Use python-dotenv, validate VENICE_API_KEY on startup
- **Vector Store**: Use Chroma for local vector DB, persist to .chroma_db directory
- **API**: Follow OpenAI chat completions format, support streaming responses