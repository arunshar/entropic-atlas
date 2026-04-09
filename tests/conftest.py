"""
Entropic Atlas — Test Configuration

Shared fixtures for all test modules.
"""

import pytest

from config import Config
from llm import LLMClient


@pytest.fixture
def config():
    """Provide a default config for testing."""
    return Config()


@pytest.fixture
def llm(config):
    """Provide an LLM client (requires API keys in env)."""
    return LLMClient(config)
