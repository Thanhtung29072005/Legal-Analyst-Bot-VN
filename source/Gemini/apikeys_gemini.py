import os
import sys

# Ensure parent directory is in path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
COHERE_API_KEY = config.COHERE_API_KEY
