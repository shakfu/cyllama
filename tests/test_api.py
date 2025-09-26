import platform

# import pytest
# pytest.skip(allow_module_level=True)

from cyllama.api import simple

PLATFORM = platform.system()
ARCH = platform.machine()

def test_api_simple(model_path):
    assert simple(
        model_path=model_path,
        prompt="When did the universe begin?",
        n_predict=32,
        n_ctx=512,
    )
    assert True
