# Cyllama Cookbook

Common patterns and recipes for using cyllama effectively.

## Table of Contents

1. [Text Generation Patterns](#text-generation-patterns)
2. [Chat Applications](#chat-applications)
3. [Structured Output](#structured-output)
4. [Performance Patterns](#performance-patterns)
5. [Integration Patterns](#integration-patterns)
6. [Error Handling](#error-handling)

## Text Generation Patterns

### Simple Question Answering

    from cyllama import LLM, GenerationConfig
    
    gen = LLM("models/llama.gguf")
    config = GenerationConfig(
    temperature=0.3,  # Low for factual responses
    max_tokens=200
    )
    
    def ask_question(question: str) -> str:
    prompt = f"Question: {question}\nAnswer:"
    return gen(prompt, config=config)
    
    print(ask_question("What is Python?"))
    print(ask_question("Who invented the telephone?"))

### Creative Writing

    from cyllama import LLM, GenerationConfig
    
    gen = LLM("models/llama.gguf")
    config = GenerationConfig(
    temperature=0.9,  # High for creativity
    top_p=0.95,
    max_tokens=500
    )
    
    def write_story(theme: str) -> str:
    prompt = f"Write a short story about {theme}:"
    return gen(prompt, config=config, stream=False)
    
    story = write_story("a robot learning to paint")
    print(story)

### Code Generation

    from cyllama import LLM, GenerationConfig
    
    gen = LLM("models/codellama.gguf")  # Use code-specific model
    config = GenerationConfig(
    temperature=0.2,  # Low for correct syntax
    max_tokens=300,
    stop_sequences=["```"]  # Stop at code fence
    )
    
    def generate_code(task: str, language: str = "python") -> str:
    prompt = f"""Write a {language} function to {task}:
    
"""
    return gen(prompt, config=config)

code = generate_code("calculate fibonacci numbers", "python")
print(code)
    
    ### Summarization
    
from cyllama import LLM, GenerationConfig

gen = LLM("models/llama.gguf")
config = GenerationConfig(
    temperature=0.3,
    max_tokens=150
)

def summarize(text: str, max_words: int = 50) -> str:
    prompt = f"""Summarize the following text in {max_words} words or less:

{text}

Summary:"""
    return gen(prompt, config=config)

long_text = """
[Your long text here...]
"""

summary = summarize(long_text)
print(summary)
    
    ## Chat Applications
    
    ### Simple Chatbot
    
from cyllama import LLM, GenerationConfig

class SimpleChatbot:
    def __init__(self, model_path: str, system_prompt: str = "You are a helpful assistant."):
        self.gen = LLM(model_path)
        self.config = GenerationConfig(temperature=0.7, max_tokens=200)
        self.history = [{"role": "system", "content": system_prompt}]

    def chat(self, user_message: str) -> str:
        # Add user message
        self.history.append({"role": "user", "content": user_message})

        # Format conversation
        prompt = self._format_history()

        # Generate response
        response = self.gen(prompt, config=self.config)

        # Add assistant response
        self.history.append({"role": "assistant", "content": response})

        return response

    def _format_history(self) -> str:
        parts = []
        for msg in self.history:
            role = msg["role"].capitalize()
            content = msg["content"]
            parts.append(f"{role}: {content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

    def clear_history(self):
        system_prompt = self.history[0] if self.history else None
        self.history = [system_prompt] if system_prompt else []

## Usage

bot = SimpleChatbot("models/llama.gguf", system_prompt="You are a Python expert.")
print(bot.chat("What is a decorator?"))
print(bot.chat("Can you show an example?"))
    
    ### Streaming Chatbot
    
from cyllama import LLM, GenerationConfig

class StreamingChatbot:
    def __init__(self, model_path: str):
        self.gen = LLM(model_path)
        self.config = GenerationConfig(temperature=0.7, max_tokens=300)

    def chat_stream(self, message: str):
        """Yield response chunks as they're generated."""
        prompt = f"User: {message}\n\nAssistant:"

        for chunk in self.gen(prompt, config=self.config, stream=True):
            yield chunk

## Usage

bot = StreamingChatbot("models/llama.gguf")

print("User: Tell me about Python")
print("Bot: ", end="")
for chunk in bot.chat_stream("Tell me about Python"):
    print(chunk, end="", flush=True)
print()
    
    ## Structured Output
    
    ### JSON Generation
    
from cyllama import LLM, GenerationConfig
import json

gen = LLM("models/llama.gguf")
config = GenerationConfig(
    temperature=0.3,
    max_tokens=200,
    stop_sequences=["}"]
)

def generate_json(prompt: str) -> dict:
    full_prompt = f"""{prompt}

Respond with valid JSON only:

{{"""

    response = gen(full_prompt, config=config)
    json_str = "{" + response + "}"

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None

## Example

result = generate_json("Create a person with name, age, and occupation")
print(json.dumps(result, indent=2))
    
    ### Constrained Generation with JSON Schema
    
from cyllama import json_schema_to_grammar, Generator, GenerationConfig

## Define JSON schema

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "email": {"type": "string"}
    },
    "required": ["name", "age", "email"]
}

## Convert to grammar

grammar = json_schema_to_grammar(schema)

gen = LLM("models/llama.gguf")

## Note: Grammar support requires additional integration

## This shows the concept; actual implementation may vary
    
    ### List Generation
    
from cyllama import LLM, GenerationConfig

gen = LLM("models/llama.gguf")
config = GenerationConfig(
    temperature=0.5,
    max_tokens=200
)

def generate_list(topic: str, count: int = 5) -> list:
    prompt = f"List {count} {topic} (one per line):\n\n1."
    response = gen(prompt, config=config)

    # Parse numbered list
    items = ["1." + response]
    for item in items[0].split("\n"):
        item = item.strip()
        if item and item[0].isdigit():
            items.append(item.split(".", 1)[1].strip())

    return items[:count]

## Example

cities = generate_list("major cities in Europe", 5)
print(cities)
    
    ## Performance Patterns
    
    ### Batch Processing with Progress
    
from tqdm import tqdm
from cyllama import BatchGenerator, GenerationConfig

def process_batch_with_progress(prompts: list, model_path: str) -> list:

## Create batch generator

    batch_gen = BatchGenerator(model_path, n_seq_max=10)
    config = GenerationConfig(max_tokens=50, temperature=0.7)

    # Split into chunks for progress tracking
    chunk_size = 10
    chunks = [prompts[i:i+chunk_size] for i in range(0, len(prompts), chunk_size)]

    results = []
    for chunk in tqdm(chunks, desc="Processing batches"):
        batch_results = batch_gen.generate_batch(chunk, config)
        results.extend(batch_results)

    return results

## Usage

prompts = [f"What is {i}+{i}?" for i in range(100)]
results = process_batch_with_progress(prompts, "models/llama.gguf")
    
    ### Parallel Generation with Threading
    
from cyllama import LLM, GenerationConfig
import concurrent.futures
import threading

## Create thread-local generators

thread_local = threading.local()

def get_generator(model_path: str) -> Generator:
    if not hasattr(thread_local, "generator"):
        thread_local.generator = Generator(model_path)
    return thread_local.generator

def generate_parallel(prompts: list, model_path: str, max_workers: int = 4) -> list:
    """Generate responses in parallel using multiple threads."""

    def process_prompt(prompt: str) -> str:
        gen = get_generator(model_path)
        config = GenerationConfig(max_tokens=100)
        return gen(prompt, config=config)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_prompt, prompts))

    return results

## Usage

prompts = ["Question 1", "Question 2", "Question 3", "Question 4"]
results = generate_parallel(prompts, "models/llama.gguf", max_workers=2)
    
    ### Memory-Efficient Generation
    
from cyllama import LLM, GenerationConfig, estimate_memory_usage

def create_memory_efficient_generator(model_path: str, available_memory_mb: int):
    """Create generator optimized for available memory."""

    # Estimate memory needs
    memory_info = estimate_memory_usage(
        model_path,
        n_ctx=2048,
        n_batch=512
    )

    # Adjust parameters if needed
    if memory_info.total_mb > available_memory_mb:
        # Reduce context
        n_ctx = 1024
        memory_info = estimate_memory_usage(model_path, n_ctx=n_ctx, n_batch=256)

    config = GenerationConfig(
        n_ctx=n_ctx if 'n_ctx' in locals() else 2048,
        n_batch=256 if memory_info.total_mb > available_memory_mb else 512,
        n_gpu_layers=0  # CPU only if memory constrained
    )

    return Generator(model_path, config=config)
    
    ## Integration Patterns
    
    ### FastAPI Server
    
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from cyllama import LLM, GenerationConfig

app = FastAPI()

## Initialize generator at startup

generator = Generator("models/llama.gguf")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 200
    temperature: float = 0.7

class GenerateResponse(BaseModel):
    text: str

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    try:
        config = GenerationConfig(
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        result = generator(request.prompt, config=config)
        return GenerateResponse(text=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

## Run with: uvicorn server:app --reload
    
    ### Flask Streaming Server
    
from flask import Flask, request, Response, stream_with_context
from cyllama import LLM, GenerationConfig
import json

app = Flask(__name__)
generator = Generator("models/llama.gguf")

@app.route("/generate-stream", methods=["POST"])
def generate_stream():
    data = request.json
    prompt = data.get("prompt", "")
    config = GenerationConfig(
        max_tokens=data.get("max_tokens", 200),
        temperature=data.get("temperature", 0.7)
    )

    def generate():
        for chunk in generator(prompt, config=config, stream=True):
            yield f"data: {json.dumps({'text': chunk})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream"
    )

## Run with: flask run
    
    ### Gradio Interface
    
import gradio as gr
from cyllama import LLM, GenerationConfig

generator = Generator("models/llama.gguf")

def generate_response(prompt, temperature, max_tokens):
    config = GenerationConfig(
        temperature=temperature,
        max_tokens=max_tokens
    )
    return generator(prompt, config=config)

interface = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Textbox(label="Prompt", lines=5),
        gr.Slider(0, 1, value=0.7, label="Temperature"),
        gr.Slider(50, 500, value=200, step=50, label="Max Tokens")
    ],
    outputs=gr.Textbox(label="Response", lines=10),
    title="Cyllama Text Generator"
)

interface.launch()
    
    ## Error Handling
    
    ### Robust Generation with Retries
    
from cyllama import LLM, GenerationConfig
import time

def generate_with_retry(
    gen: Generator,
    prompt: str,
    config: GenerationConfig,
    max_retries: int = 3
) -> str:
    """Generate with exponential backoff retry."""

    for attempt in range(max_retries):
        try:
            return gen(prompt, config=config)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt
            print(f"Error: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)

## Usage

gen = LLM("models/llama.gguf")
config = GenerationConfig(max_tokens=100)

try:
    result = generate_with_retry(gen, "Test prompt", config)
    print(result)
except Exception as e:
    print(f"Failed after retries: {e}")
    
    ### Timeout Handling
    
from cyllama import LLM, GenerationConfig
import signal
from contextlib import contextmanager

class TimeoutError(Exception):
    pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def generate_with_timeout(prompt: str, timeout_seconds: int = 30) -> str:
    gen = LLM("models/llama.gguf")
    config = GenerationConfig(max_tokens=200)

    try:
        with time_limit(timeout_seconds):
            return gen(prompt, config=config)
    except TimeoutError:
        return "Generation timed out"

## Usage

result = generate_with_timeout("Generate a very long essay", timeout_seconds=10)
    
    ### Validation and Sanitization
    
from cyllama import LLM, GenerationConfig
import re

def safe_generate(prompt: str, model_path: str) -> str:
    """Generate with input validation and output sanitization."""

    # Validate input
    if not prompt or len(prompt) < 3:
        raise ValueError("Prompt too short")

    if len(prompt) > 1000:
        raise ValueError("Prompt too long")

    # Sanitize prompt
    prompt = prompt.strip()
    prompt = re.sub(r'\s+', ' ', prompt)  # Normalize whitespace

    # Generate
    gen = LLM(model_path)
    config = GenerationConfig(
        max_tokens=200,
        temperature=0.7
    )

    response = gen(prompt, config=config)

    # Sanitize response
    response = response.strip()
    response = re.sub(r'\n{3,}', '\n\n', response)  # Max 2 newlines

    return response

## Usage

try:
    result = safe_generate("What is AI?", "models/llama.gguf")
    print(result)
except ValueError as e:
    print(f"Invalid input: {e}")
    
    ## Tips and Tricks
    
    ### Prompt Engineering
    

## Be specific

prompt = "Write a Python function that takes a list of integers and returns the sum"

## Use examples (few-shot)

prompt = """
Q: What is 2+2?
A: 4

Q: What is 3+3?
A: 6

Q: What is 5+5?
A:"""

## Set context

prompt = "You are an expert Python programmer. Explain list comprehensions in simple terms:"
    
    ### Response Post-Processing
    
def clean_response(response: str) -> str:
    """Clean up generated response."""

## Remove common artifacts

    response = response.strip()
    response = response.replace("</s>", "")
    response = response.replace("<|endoftext|>", "")

    # Remove incomplete sentences
    sentences = response.split(".")
    if sentences[-1] and not sentences[-1].endswith("?"):
        sentences = sentences[:-1]

    return ".".join(sentences) + "."
    
    ## See Also
    
    - [User Guide](USER_GUIDE.md) - Complete API documentation
    - [Examples](../tests/examples/) - Working code samples
    - [API Reference](API_REFERENCE.md) - Detailed API docs
```
