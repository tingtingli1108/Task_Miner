"""
LLM Model Management System for TaskMiner

This module provides a centralized system for managing and instantiating
different OpenAI LLM models, with extensible design for future providers.
Updated with the latest OpenAI models including o3 mini and o3 series.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

# Import LangChain LLM classes
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.llms.base import LLM

# Import pandas for DataFrame functionality
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# Load environment variables
load_dotenv()
logger = logging.getLogger(__name__)

class ModelProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    # Future providers can be added here:
    # GOOGLE = "google"

@dataclass
class ModelConfig:
    """Configuration for an LLM model"""
    provider: ModelProvider
    model_name: str
    display_name: str
    description: str
    max_tokens: Optional[int] = None
    temperature: float = 0
    cost_per_1m_input_tokens: Optional[float] = None
    cost_per_1m_output_tokens: Optional[float] = None
    supports_streaming: bool = True
    context_window: Optional[int] = None
    api_key_env_var: str = "OPENAI_API_KEY"
    additional_params: Dict[str, Any] = None
    is_multimodal: bool = False
    is_reasoning: bool = False
    release_date: Optional[str] = None

class LLMModelRegistry:
    """Registry of all available LLM models"""
    
    def __init__(self):
        self.models: Dict[str, ModelConfig] = {}
        self._register_openai_models()
        self._register_anthropic_models()
    
    def _register_anthropic_models(self):
        """Register all current Anthropic models"""
        
        anthropic_models = [
            # Latest Claude 4 Models (2025)
            ModelConfig(
                provider=ModelProvider.ANTHROPIC,
                model_name="claude-opus-4-20250514",
                display_name="Claude Opus 4",
                description="Successor to Claude 3 Opus, Anthropic's most powerful model for complex tasks.",
                max_tokens=4096,
                temperature=0,
                cost_per_1m_input_tokens=15.00,
                cost_per_1m_output_tokens=75.00,
                context_window=200000,
                api_key_env_var="ANTHROPIC_API_KEY",
                is_multimodal=True,
                release_date="2025-05-14"
            ),
            ModelConfig(
                provider=ModelProvider.ANTHROPIC,
                model_name="claude-sonnet-4-20250514",
                display_name="Claude Sonnet 4",
                description="Successor to Claude 3.5 Sonnet, offering a great balance of intelligence and speed.",
                max_tokens=4096,
                temperature=0,
                cost_per_1m_input_tokens=3.00,
                cost_per_1m_output_tokens=15.00,
                context_window=200000,
                api_key_env_var="ANTHROPIC_API_KEY",
                is_multimodal=True,
                release_date="2025-05-14"
            ),
            ModelConfig(
                provider=ModelProvider.ANTHROPIC,
                model_name="claude-3-7-sonnet-20250219",
                display_name="Claude Sonnet 3.7",
                description="An iteration on the Sonnet family, offering strong performance.",
                max_tokens=4096,
                temperature=0,
                cost_per_1m_input_tokens=3.00,
                cost_per_1m_output_tokens=15.00,
                context_window=200000,
                api_key_env_var="ANTHROPIC_API_KEY",
                is_multimodal=True,
                release_date="2025-02-19"
            ),
            ModelConfig(
                provider=ModelProvider.ANTHROPIC,
                model_name="claude-3-5-haiku-20241022",
                display_name="Claude Haiku 3.5",
                description="An iteration on the Haiku family, fastest and most compact model.",
                max_tokens=4096,
                temperature=0,
                cost_per_1m_input_tokens=0.80,
                cost_per_1m_output_tokens=4.00,
                context_window=200000,
                api_key_env_var="ANTHROPIC_API_KEY",
                is_multimodal=True,
                release_date="2024-10-22"
            ),

            # # Claude 3 Models
            # ModelConfig(
            #     provider=ModelProvider.ANTHROPIC,
            #     model_name="claude-3-opus-20240229",
            #     display_name="Claude 3 Opus",
            #     description="Anthropic's powerful model for a wide range of complex tasks.",
            #     max_tokens=4096,
            #     temperature=0,
            #     cost_per_1m_input_tokens=15.00,
            #     cost_per_1m_output_tokens=75.00,
            #     context_window=200000,
            #     api_key_env_var="ANTHROPIC_API_KEY",
            #     is_multimodal=True,
            #     release_date="2024-02-29"
            # ),
            # ModelConfig(
            #     provider=ModelProvider.ANTHROPIC,
            #     model_name="claude-3-haiku-20240307",
            #     display_name="Claude 3 Haiku",
            #     description="Anthropic's fastest, most compact model for near-instant responsiveness.",
            #     max_tokens=4096,
            #     temperature=0,
            #     cost_per_1m_input_tokens=0.25,
            #     cost_per_1m_output_tokens=1.25,
            #     context_window=200000,
            #     api_key_env_var="ANTHROPIC_API_KEY",
            #     is_multimodal=True,
            #     release_date="2024-03-07"
            # ),
        ]
        
        # Register all models
        for model in anthropic_models:
            self.register_model(model)

    def _register_openai_models(self):
        """Register all current OpenAI models"""
        
        openai_models = [
            # Latest o3 Reasoning Models (2025)
            ModelConfig(
                provider=ModelProvider.OPENAI,
                model_name="o3-mini",
                display_name="o3 Mini",
                description="Latest reasoning model, specialized for technical domains requiring precision and speed. Available to all users including free tier.",
                max_tokens=100000,
                temperature=0,
                cost_per_1m_input_tokens=1.10,
                cost_per_1m_output_tokens=4.40,
                context_window=200000,
                is_reasoning=True,
                is_multimodal=False,
                release_date="2025-01-31"
            ),
            ModelConfig(
                provider=ModelProvider.OPENAI,
                model_name="o3",
                display_name="o3",
                description="Advanced reasoning model with significantly better performance on complex tasks including coding, mathematics, and science",
                max_tokens=100000,
                temperature=0,
                cost_per_1m_input_tokens=15.00,
                cost_per_1m_output_tokens=60.00,
                context_window=200000,
                is_reasoning=True,
                is_multimodal=False,
                release_date="2025-04-16"
            ),
            ModelConfig(
                provider=ModelProvider.OPENAI,
                model_name="o3-pro",
                display_name="o3 Pro",
                description="OpenAI's most capable model yet. Recommended for challenging questions where reliability matters more than speed",
                max_tokens=100000,
                temperature=0,
                cost_per_1m_input_tokens=50.00,
                cost_per_1m_output_tokens=150.00,
                context_window=200000,
                is_reasoning=True,
                is_multimodal=False,
                release_date="2025-06-10"
            ),
            ModelConfig(
                provider=ModelProvider.OPENAI,
                model_name="o4-mini",
                display_name="o4 Mini",
                description="Successor to o3-mini with enhanced capabilities",
                max_tokens=100000,
                temperature=0,
                cost_per_1m_input_tokens=1.50,
                cost_per_1m_output_tokens=5.00,
                context_window=200000,
                is_reasoning=True,
                is_multimodal=False,
                release_date="2025-04-16"
            ),
            
            # Latest GPT-4o models
            ModelConfig(
                provider=ModelProvider.OPENAI,
                model_name="gpt-4o",
                display_name="GPT-4o",
                description="Latest flagship model, multimodal with vision capabilities, faster and cheaper than GPT-4 Turbo",
                max_tokens=4096,
                temperature=0,
                cost_per_1m_input_tokens=5.00,
                cost_per_1m_output_tokens=15.00,
                context_window=128000,
                is_multimodal=True,
                release_date="2024-05"
            ),
            ModelConfig(
                provider=ModelProvider.OPENAI,
                model_name="gpt-4o-mini",
                display_name="GPT-4o Mini",
                description="Smaller, faster, and cheaper version of GPT-4o, great for most tasks",
                max_tokens=4096,
                temperature=0,
                cost_per_1m_input_tokens=0.15,
                cost_per_1m_output_tokens=0.60,
                context_window=128000,
                is_multimodal=True,
                release_date="2024-07"
            ),
            
            # Latest GPT-4 models
            ModelConfig(
                provider=ModelProvider.OPENAI,
                model_name="gpt-4.5",
                display_name="GPT-4.5",
                description="Latest evolution of GPT-4 with enhanced capabilities",
                max_tokens=4096,
                temperature=0,
                cost_per_1m_input_tokens=20.00,
                cost_per_1m_output_tokens=40.00,
                context_window=128000,
                is_multimodal=False,
                release_date="2025"
            ),
            ModelConfig(
                provider=ModelProvider.OPENAI,
                model_name="gpt-4.1",
                display_name="GPT-4.1 (Default)",
                description="Current production model - enhanced version of GPT-4",
                max_tokens=4096,
                temperature=0,
                cost_per_1m_input_tokens=30.00,
                cost_per_1m_output_tokens=60.00,
                context_window=8192,
                is_multimodal=False,
                release_date="2023"
            ),
            
            # GPT-4 Turbo models
            ModelConfig(
                provider=ModelProvider.OPENAI,
                model_name="gpt-4-turbo",
                display_name="GPT-4 Turbo",
                description="Previous generation flagship model with vision capabilities",
                max_tokens=4096,
                temperature=0,
                cost_per_1m_input_tokens=10.00,
                cost_per_1m_output_tokens=30.00,
                context_window=128000,
                is_multimodal=True,
                release_date="2024-04"
            ),
            
            # Standard GPT-4 models
            ModelConfig(
                provider=ModelProvider.OPENAI,
                model_name="gpt-4",
                display_name="GPT-4",
                description="Original GPT-4, highest quality reasoning for complex tasks",
                max_tokens=4096,
                temperature=0,
                cost_per_1m_input_tokens=30.00,
                cost_per_1m_output_tokens=60.00,
                context_window=8192,
                is_multimodal=False,
                release_date="2023-03"
            ),
            
            # o1 reasoning models (previous generation)
            ModelConfig(
                provider=ModelProvider.OPENAI,
                model_name="o1-preview",
                display_name="o1 Preview",
                description="Previous generation reasoning model (preview)",
                max_tokens=32768,
                temperature=0,
                cost_per_1m_input_tokens=15.00,
                cost_per_1m_output_tokens=60.00,
                context_window=128000,
                is_reasoning=True,
                is_multimodal=False,
                release_date="2024-09"
            )
        ]
        
        # Register all models
        for model in openai_models:
            self.register_model(model)
    
    def register_model(self, config: ModelConfig):
        """Register a new model configuration"""
        model_id = f"{config.provider.value}:{config.model_name}"
        self.models[model_id] = config
        logger.debug(f"Registered model: {model_id}")
    
    def get_model_config(self, model_id: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model"""
        return self.models.get(model_id)
    
    def list_models(self, provider: Optional[ModelProvider] = None) -> Dict[str, ModelConfig]:
        """List all available models, optionally filtered by provider"""
        if provider:
            return {
                model_id: config for model_id, config in self.models.items()
                if config.provider == provider
            }
        return self.models.copy()
    
    def list_available_models(self) -> Dict[str, ModelConfig]:
        """List only models that have their API keys available"""
        available = {}
        for model_id, config in self.models.items():
            api_key = os.getenv(config.api_key_env_var)
            if api_key:
                available[model_id] = config
            else:
                logger.debug(f"Model {model_id} not available - missing {config.api_key_env_var}")
        return available
    
    def get_models_by_provider(self) -> Dict[str, List[str]]:
        """Get models grouped by provider"""
        by_provider = {}
        for model_id, config in self.models.items():
            provider = config.provider.value
            if provider not in by_provider:
                by_provider[provider] = []
            by_provider[provider].append(model_id)
        return by_provider
    
    def to_dataframe(self, available_only: bool = True) -> 'pd.DataFrame':
        """Convert models to a pandas DataFrame for easy analysis and display"""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for DataFrame functionality. Install with: pip install pandas")
        
        models = self.list_available_models() if available_only else self.models
        
        if not models:
            return pd.DataFrame()
        
        # Convert models to list of dictionaries
        data = []
        for model_id, config in models.items():
            row = {
                'Model ID': model_id,
                'Display Name': config.display_name,
                # 'Model Name': config.model_name,
                # 'Provider': config.provider.value,
                # 'Description': config.description,
                'Context Window': config.context_window,
                # 'Max Tokens': config.max_tokens,
                'Input Cost ($/1M)': config.cost_per_1m_input_tokens,
                'Output Cost ($/1M)': config.cost_per_1m_output_tokens,
                # 'Temperature': config.temperature,
                # 'Multimodal': config.is_multimodal,
                'Reasoning': config.is_reasoning,
                # 'Streaming': config.supports_streaming,
                'Release Date': config.release_date,
                # 'API Key Env': config.api_key_env_var
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Sort by release date (newest first), then by input cost
        df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')
        df = df.sort_values(['Release Date', 'Input Cost ($/1M)'], ascending=[False, True])
        
        return df
    
    def display_models_dataframe(self, available_only: bool = True, style: bool = True) -> 'pd.DataFrame':
        """Display models as a formatted pandas DataFrame"""
        df = self.to_dataframe(available_only)
        
        if df.empty:
            print("No models available. Please check your API keys.")
            return df
        
        if style and PANDAS_AVAILABLE:
            # Create a styled version for better display
            styled_df = df.copy()
            
            # Format numeric columns
            if 'Input Cost ($/1M)' in styled_df.columns:
                styled_df['Input Cost ($/1M)'] = styled_df['Input Cost ($/1M)'].apply(
                    lambda x: f"${x:.2f}" if pd.notna(x) else "N/A"
                )
            if 'Output Cost ($/1M)' in styled_df.columns:
                styled_df['Output Cost ($/1M)'] = styled_df['Output Cost ($/1M)'].apply(
                    lambda x: f"${x:.2f}" if pd.notna(x) else "N/A"
                )
            if 'Context Window' in styled_df.columns:
                styled_df['Context Window'] = styled_df['Context Window'].apply(
                    lambda x: f"{x:,}" if pd.notna(x) else "N/A"
                )
            if 'Max Tokens' in styled_df.columns:
                styled_df['Max Tokens'] = styled_df['Max Tokens'].apply(
                    lambda x: f"{x:,}" if pd.notna(x) else "N/A"
                )
            
            # Format boolean columns
            #styled_df['Multimodal'] = styled_df['Multimodal'].map({True: '✓', False: '✗'})
            styled_df['Reasoning'] = styled_df['Reasoning'].map({True: '✓', False: '✗'})
            #styled_df['Streaming'] = styled_df['Streaming'].map({True: '✓', False: '✗'})
            
            # Format dates
            styled_df['Release Date'] = styled_df['Release Date'].dt.strftime('%Y-%m')
            styled_df['Release Date'] = styled_df['Release Date'].fillna('N/A')
            
            return styled_df
        
        return df

class LLMFactory:
    """Factory for creating LLM instances"""
    
    def __init__(self, registry: LLMModelRegistry):
        self.registry = registry
    
    def create_llm(
        self, 
        model_id: str, 
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLM:
        """
        Create an LLM instance for the specified model
        
        Args:
            model_id: Model identifier (e.g., "openai:o3-mini", "openai:gpt-4o")
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            **kwargs: Additional model-specific parameters
            
        Returns:
            Configured LLM instance
            
        Raises:
            ValueError: If model not found or API key not available
        """
        config = self.registry.get_model_config(model_id)
        if not config:
            available_models = list(self.registry.models.keys())
            raise ValueError(f"Model '{model_id}' not found. Available models: {available_models}")
        
        # Check if API key is available
        api_key = os.getenv(config.api_key_env_var)
        if not api_key:
            raise ValueError(f"API key '{config.api_key_env_var}' not found in environment variables")
        
        # Use provided parameters or fall back to config defaults
        final_temperature = temperature if temperature is not None else config.temperature
        final_max_tokens = max_tokens if max_tokens is not None else config.max_tokens
        
        # Start with basic parameters
        params = {
            "api_key": api_key,
            **kwargs
        }
        
        # Only add temperature for models that support it (non-reasoning models)
        if not config.is_reasoning:
            params["temperature"] = final_temperature
        
        if final_max_tokens:
            params["max_tokens"] = final_max_tokens
        
        # Add any model-specific parameters
        if config.additional_params:
            params.update(config.additional_params)
        
        # Create the appropriate LLM instance
        if config.provider == ModelProvider.OPENAI:
            # Use model_name instead of model for ChatOpenAI
            return ChatOpenAI(model=config.model_name, **params)
        elif config.provider == ModelProvider.ANTHROPIC:
            return ChatAnthropic(model=config.model_name, **params)  # Use ChatAnthropic, not LLM
        else:
            # Future providers will be handled here
            raise ValueError(f"Unsupported provider: {config.provider}")

# Global instances
model_registry = LLMModelRegistry()
llm_factory = LLMFactory(model_registry)

def get_available_models() -> Dict[str, ModelConfig]:
    """Get all models that have API keys available"""
    return model_registry.list_available_models()

def create_llm(model_id: str, **kwargs) -> LLM:
    """Convenience function to create an LLM instance"""
    return llm_factory.create_llm(model_id, **kwargs)

def get_default_model() -> str:
    """Get the default model ID with preference for latest models"""
    available = get_available_models()
    
    # Priority order for default model (latest first)
    candidates = [
        "openai:gpt-4.1",      # Your current production model (supports temperature)  
        "openai:gpt-4o",       # Latest multimodal model
        "openai:o3-mini",      # Latest reasoning model, available to all users
        "openai:o4-mini",      # Successor to o3-mini
        "openai:gpt-4.5",      # Enhanced GPT-4
        "openai:gpt-4-turbo",  # Good fallback
        "openai:gpt-4",        # Standard fallback
        # "openai:gpt-3.5-turbo", # Last resort
        "anthropic:claude-sonnet-4-20250514", # Latest model (medium)
        "anthropic:claude-opus-4-20250514", # Latest model (expensive)
        "anthropic:claude-3-7-sonnet-20250219", # Second latest model
    ]
    
    for candidate in candidates:
        if candidate in available:
            return candidate
    
    # If no models available, raise error
    if not available:
        raise ValueError("No models available. Please check your PROVIDER_API_KEY environment variables.")
    
    # Return first available model as last resort
    return list(available.keys())[0] 

def display_models_table(available_only: bool = True) -> str:
    """Convenience function to display the models table"""
    return model_registry.display_models_table(available_only)

def get_models_dataframe(available_only: bool = True) -> 'pd.DataFrame':
    """Convenience function to get models as a DataFrame"""
    return model_registry.to_dataframe(available_only)

def display_models_dataframe(available_only: bool = True, style: bool = True) -> 'pd.DataFrame':
    """Convenience function to display models as a formatted DataFrame"""
    return model_registry.display_models_dataframe(available_only, style)

# Example usage for displaying the table
if __name__ == "__main__":
    print("Available Models:")
    print("=" * 120)
    print(display_models_dataframe(available_only=True))