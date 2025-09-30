"""
Configuration module for Multi-Source RAG Engine.

This module loads environment variables from a .env file and provides
them as constants for other parts of the application. It includes validation
to ensure that critical configuration, like API keys, are present.
"""

import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = "gemini-2.5-flash"  # Model name is now configurable

if not GEMINI_API_KEY:
    raise ValueError(
        "Gemini API key not found. "
        "Please create a .env file in the project root and add the following line:\n"
        "GEMINI_API_KEY='your-api-key-here'"
    )