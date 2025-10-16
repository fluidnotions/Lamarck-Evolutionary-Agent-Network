"""Pytest configuration and shared fixtures."""

import pytest
import logging

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@pytest.fixture
def sample_validation_request() -> dict:
    """Sample validation request for testing."""
    return {
        "type": "user_registration",
        "description": "Validate user registration data",
        "requirements": ["schema", "business_rules"]
    }


@pytest.fixture
def sample_input_data() -> dict:
    """Sample input data for testing."""
    return {
        "user": {
            "username": "testuser",
            "email": "test@example.com",
            "age": 25,
            "country": "US"
        }
    }
