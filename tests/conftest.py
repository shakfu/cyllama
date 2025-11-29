import sys
from pathlib import Path
ROOT = Path.cwd()

# Default model path constant (for use in subprocesses where fixtures aren't available)
DEFAULT_MODEL = "models/Llama-3.2-1B-Instruct-Q8_0.gguf"

import pytest

@pytest.fixture(scope="module")
def model_path():
	return str(ROOT / DEFAULT_MODEL)
