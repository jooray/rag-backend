from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name: str = Field(description="Model name to use for Venice API")
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=None)


class PromptConfig(BaseModel):
    system_prompt: str = Field(description="System prompt for the model")
    user_prompt_template: str = Field(
        description="User prompt template, use {question} and {context} placeholders"
    )
    model: str = Field(description="Model ID to use from models config")


class GatePromptConfig(BaseModel):
    name: str = Field(description="Name of the gate prompt")
    system_prompt: str = Field(description="System prompt for gate checking")
    user_prompt_template: str = Field(
        description="Template for gate check, use {response} placeholder"
    )
    model: str = Field(description="Model ID to use from models config")
    fix_prompt: Optional["FixPromptConfig"] = None


class FixPromptConfig(BaseModel):
    system_prompt: str = Field(description="System prompt for fixing responses")
    user_prompt_template: str = Field(
        description="Template for fixing, use {response} and {reject_reason} placeholders"
    )
    model: str = Field(description="Model ID to use from models config")


class RewritePromptConfig(BaseModel):
    name: str = Field(description="Name of the rewrite prompt")
    system_prompt: str = Field(description="System prompt for rewriting")
    user_prompt_template: str = Field(
        description="Template for rewriting, use {response} placeholder"
    )
    model: str = Field(description="Model ID to use from models config")


class PipelineConfig(BaseModel):
    main_prompt: PromptConfig
    gate_prompts: List[GatePromptConfig] = Field(default_factory=list)
    rewrite_prompts: List[RewritePromptConfig] = Field(default_factory=list)
    max_retries: int = Field(default=2, ge=1, le=10)


class VectorDBConfig(BaseModel):
    collection_name: str = Field(default="rag_documents")
    embedding_model: str = Field(default="nomic-embed-text")
    ollama_base_url: str = Field(default="http://localhost:11434")
    chunk_size: int = Field(default=500)
    chunk_overlap: int = Field(default=50)
    top_k: int = Field(default=5)
    use_mmr: bool = Field(default=False, description="Use Maximal Marginal Relevance for diverse results")
    mmr_fetch_k: int = Field(default=10, description="Number of documents to fetch before MMR reranking")
    mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0, description="MMR lambda parameter (0=diversity, 1=relevance)")


class CORSConfig(BaseModel):
    origins: List[str] = Field(default_factory=list, description="Allowed origins for CORS")
    methods: List[str] = Field(default=["GET", "POST", "PUT", "DELETE", "OPTIONS"], description="Allowed HTTP methods")
    headers: List[str] = Field(default=["Content-Type", "Authorization"], description="Allowed headers")
    supports_credentials: bool = Field(default=False, description="Allow credentials in CORS requests")


class ServerConfig(BaseModel):
    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: int = Field(default=8080, ge=1, le=65535, description="Port to bind to")
    debug: bool = Field(default=False, description="Enable debug mode")
    cors: CORSConfig = Field(default_factory=CORSConfig, description="CORS configuration")


class Config(BaseModel):
    venice_api_base: str = Field(default="https://api.venice.ai/api/v1")
    data_directory: str = Field(default="data")
    vector_db_config: VectorDBConfig
    pipeline_config: PipelineConfig
    server_config: ServerConfig = Field(default_factory=ServerConfig)
    models: Dict[str, ModelConfig] = Field(default_factory=dict)


GatePromptConfig.model_rebuild()
