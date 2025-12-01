# import pytest
# pytest.skip(allow_module_level=True)

import cyllama.llama.llama_cpp as cy
from cyllama import apply_chat_template, get_chat_template, LLM, Response


def test_chat_builtin_templates():
    assert cy.chat_builtin_templates() == [
    'bailing',
    'bailing-think',
    'bailing2',
    'chatglm3',
    'chatglm4',
    'chatml',
    'command-r',
    'deepseek',
    'deepseek2',
    'deepseek3',
    'exaone3',
    'exaone4',
    'falcon3',
    'gemma',
    'gigachat',
    'glmedge',
    'gpt-oss',
    'granite',
    'grok-2',
    'hunyuan-dense',
    'hunyuan-moe',
    'kimi-k2',
    'llama2',
    'llama2-sys',
    'llama2-sys-bos',
    'llama2-sys-strip',
    'llama3',
    'llama4',
    'megrez',
    'minicpm',
    'mistral-v1',
    'mistral-v3',
    'mistral-v3-tekken',
    'mistral-v7',
    'mistral-v7-tekken',
    'monarch',
    'openchat',
    'orion',
    'pangu-embedded',
    'phi3',
    'phi4',
    'rwkv-world',
    'seed_oss',
    'smolvlm',
    'vicuna',
    'vicuna-orca',
    'yandex',
    'zephyr',
]


def test_get_chat_template(model_path):
    """Test that get_chat_template retrieves a template from the model."""
    template = get_chat_template(model_path)
    # Llama 3.2 models should have a chat template
    assert isinstance(template, str)
    # The template should contain some typical template markers
    assert len(template) > 0


def test_get_chat_template_by_name(model_path):
    """Test retrieving a specific template by name.

    Note: get_chat_template_by_name retrieves templates stored in the model
    metadata, not the builtin templates list. Most models only have one
    default template. This test verifies the function returns a string
    (which may be empty if the named template doesn't exist).
    """
    # The default template (None) should exist for Llama 3.2 models
    template = get_chat_template(model_path, template_name=None)
    assert isinstance(template, str)
    assert len(template) > 0

    # A non-existent named template should return empty string
    template = get_chat_template(model_path, template_name="nonexistent")
    assert isinstance(template, str)
    assert template == ""


def test_apply_chat_template_basic(model_path):
    """Test basic chat template application."""
    messages = [
        {"role": "user", "content": "Hello!"}
    ]
    prompt = apply_chat_template(messages, model_path)
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    # The prompt should contain the user content
    assert "Hello" in prompt


def test_apply_chat_template_with_system(model_path):
    """Test chat template with system message."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"}
    ]
    prompt = apply_chat_template(messages, model_path)
    assert isinstance(prompt, str)
    # Both messages should be in the formatted prompt
    assert "helpful assistant" in prompt
    assert "Python" in prompt


def test_apply_chat_template_with_explicit_template(model_path):
    """Test applying a specific template by name."""
    messages = [
        {"role": "user", "content": "Hello!"}
    ]
    prompt = apply_chat_template(messages, model_path, template="llama3")
    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_apply_chat_template_multi_turn(model_path):
    """Test multi-turn conversation formatting."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello! How can I help?"},
        {"role": "user", "content": "What is 2+2?"}
    ]
    prompt = apply_chat_template(messages, model_path)
    assert isinstance(prompt, str)
    # All messages should be present
    assert "helpful" in prompt
    assert "Hi" in prompt
    assert "Hello" in prompt
    assert "2+2" in prompt


def test_llm_get_chat_template(model_path):
    """Test LLM.get_chat_template method."""
    with LLM(model_path) as llm:
        template = llm.get_chat_template()
        assert isinstance(template, str)
        assert len(template) > 0


def test_llm_chat_method(model_path):
    """Test LLM.chat method for chat-style generation."""
    messages = [
        {"role": "user", "content": "Say 'test' and nothing else."}
    ]
    with LLM(model_path, max_tokens=10) as llm:
        response = llm.chat(messages)
        assert isinstance(response, Response)
        assert len(response) > 0

