# Security Policy

## Reporting a vulnerability

Report vulnerabilities privately through [GitHub private vulnerability reporting](https://github.com/shakfu/cyllama/security/advisories/new). Do not open a public issue.

Include the cyllama version, platform and wheel variant (`cyllama info`), and steps to reproduce.

## Supported versions

Fixes land in the latest release only.

## Scope

In scope:

- The Cython bindings and helper C/C++ code under `src/cyllama/`.
- The OpenAI-compatible servers, `EmbeddedServer` and `PythonServer`.
- Wheel packaging, including bundled shared libraries.

Report vulnerabilities in llama.cpp, whisper.cpp, stable-diffusion.cpp, Mongoose or sqlite-vector upstream. Tell us too if a cyllama release ships the affected version.

## Deployment notes

- Both servers bind `127.0.0.1` by default and have no TLS.
- Binding another address without an API key lets anyone who can reach the port use the model. See [Authentication and Limits](docs/server_usage_examples.md#authentication-and-limits).
- Model files (GGUF, safetensors) are parsed by native code. Load models only from sources you trust.
