import platform

import pytest
pytest.skip(allow_module_level=True)

from cyllama import Llama

PLATFORM = platform.system()
ARCH = platform.machine()


@pytest.mark.skipif(PLATFORM == "Darwin" and ARCH == "x86_64", reason="Skip on intel macs")
def test_api(model_path):
    llm = Llama(model_path=model_path, disable_log=True, n_predict=32, n_ctx=512)
    prompt = "What is 2 * 10?"
    assert "2 * 10 = 20\n" in llm.ask(prompt)

@pytest.mark.skipif(PLATFORM == "Darwin" and ARCH == "x86_64", reason="Skip on intel macs")
def test_ask_answer(model_path):
    llm = Llama(model_path=model_path, disable_log=True, n_predict=128, n_ctx=512)
    prompt = "When did the universe begin?"
    result = llm.ask(prompt)
    assert "around 13.8 billion years ago" in result
