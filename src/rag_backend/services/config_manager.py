from typing import Dict, Optional
import os

from ..models.config import Config, ConfigurationEntry
from .vector_db import VectorDBService
from .pipeline import PipelineService


class ConfigurationManager:
    """Manages multiple configurations and their associated services"""

    def __init__(self, config: Config, api_key: str, reindex: bool = False):
        self.config = config
        self.api_key = api_key
        self.vector_db_services: Dict[str, VectorDBService] = {}
        self.pipeline_services: Dict[str, PipelineService] = {}

        # Initialize all configurations
        self._initialize_configurations(reindex)

    def _initialize_configurations(self, reindex: bool = False):
        """Initialize vector DB and pipeline services for all configurations"""
        for model_name, config_entry in self.config.configurations.items():
            # Initialize vector DB service
            vector_db = VectorDBService(
                config=config_entry.vector_db_config,
                data_dir=config_entry.data_directory,
            )
            vector_db.load_or_create_index(reindex=reindex)
            self.vector_db_services[model_name] = vector_db

            # Initialize pipeline service
            pipeline = PipelineService(
                config=config_entry.pipeline_config,
                models=self.config.models,
                api_key=self.api_key,
                api_base=self.config.venice_api_base
            )
            self.pipeline_services[model_name] = pipeline

    def get_vector_db_service(self, model_name: str) -> Optional[VectorDBService]:
        """Get vector DB service for a specific model configuration"""
        return self.vector_db_services.get(model_name)

    def get_pipeline_service(self, model_name: str) -> Optional[PipelineService]:
        """Get pipeline service for a specific model configuration"""
        return self.pipeline_services.get(model_name)

    def get_available_models(self) -> list[str]:
        """Get list of available model configurations"""
        return list(self.config.configurations.keys())

    def has_configuration(self, model_name: str) -> bool:
        """Check if a configuration exists for the given model name"""
        return model_name in self.config.configurations
