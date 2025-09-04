import sys
from pathlib import Path
ROOT = Path.cwd()


import pytest

@pytest.fixture(scope="module")
def model_path():
	return str(ROOT / 'models' / 'gemma-3-270m-it-Q5_K_S.gguf')
