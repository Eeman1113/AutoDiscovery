"""
Configuration settings for the Multi-Agent Autonomous Research System
"""
import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """System configuration class"""
    
    # Gemini API Configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = "gemini-pro"
    GEMINI_MAX_TOKENS = 8192
    GEMINI_TEMPERATURE = 0.7
    
    # System Configuration
    MAX_RESEARCH_ITERATIONS = 5
    MAX_PARALLEL_AGENTS = 4
    RESEARCH_TIMEOUT = 300  # seconds
    VALIDATION_CONFIDENCE_THRESHOLD = 0.8
    
    # Agent Configuration
    AGENT_CONFIGS = {
        "coordinator": {
            "max_retries": 3,
            "timeout": 60,
            "confidence_threshold": 0.7
        },
        "research": {
            "max_sources": 20,
            "min_source_quality": 0.6,
            "search_depth": "comprehensive"
        },
        "analysis": {
            "analysis_depth": "deep",
            "correlation_threshold": 0.5,
            "max_patterns": 10
        },
        "creative": {
            "creativity_level": "high",
            "max_alternatives": 5,
            "innovation_threshold": 0.8
        },
        "validation": {
            "validation_layers": ["factual", "logical", "source", "bias"],
            "min_confidence": 0.8,
            "max_contradictions": 2
        },
        "documentation": {
            "output_formats": ["markdown", "pdf", "html"],
            "include_visualizations": True,
            "citation_style": "academic"
        }
    }
    
    # Quality Control Settings
    QUALITY_METRICS = {
        "comprehensiveness_threshold": 0.8,
        "accuracy_threshold": 0.9,
        "originality_threshold": 0.7,
        "actionability_threshold": 0.8
    }
    
    # Communication Settings
    MESSAGE_FORMAT = {
        "include_metadata": True,
        "include_confidence_scores": True,
        "include_source_references": True
    }
    
    # Logging Configuration
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def get_agent_config(cls, agent_name: str) -> Dict[str, Any]:
        """Get configuration for a specific agent"""
        return cls.AGENT_CONFIGS.get(agent_name, {})
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that all required configuration is present"""
        if not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required in environment variables")
        return True 