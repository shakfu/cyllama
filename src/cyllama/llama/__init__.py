# Lazy imports to avoid circular dependency:
# __init__ -> server/python.py -> llama_cpp (extension not yet registered)

_lazy_imports = {
    "LlamaCLI": (".cli", "LlamaCLI"),
    "ServerConfig": (".server", "ServerConfig"),
    "LlamaServer": (".server", "LlamaServer"),
    "LlamaServerClient": (".server", "LlamaServerClient"),
    "start_server": (".server", "start_server"),
}

def __getattr__(name):
    if name in _lazy_imports:
        module_path, attr = _lazy_imports[name]
        import importlib
        mod = importlib.import_module(module_path, __name__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
