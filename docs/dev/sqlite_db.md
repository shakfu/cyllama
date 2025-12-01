# SQLite Database Design for cyllama

## Overview

This document outlines the design for an optional SQLite-based storage system for cyllama, providing:

1. **Response Caching** - Cache responses for identical prompts with TTL
2. **Statistics Tracking** - Track generation metrics over time
3. **Configuration Storage** - Persistent user preferences and model defaults
4. **Structured Logging** - Queryable generation logs

## Design Principles

- **Opt-in by default** - Zero overhead unless explicitly enabled
- **Single file** - All data in one SQLite database file
- **Thread-safe** - WAL mode for concurrent read/write
- **Zero dependencies** - Uses Python's built-in `sqlite3` module
- **Graceful degradation** - Works without DB (in-memory fallback)

## Database Location

Default location follows XDG Base Directory Specification:

```
~/.local/share/cyllama/cyllama.db      # Linux
~/Library/Application Support/cyllama/cyllama.db  # macOS
%LOCALAPPDATA%\cyllama\cyllama.db      # Windows
```

Can be overridden via:
- Environment variable: `CYLLAMA_DB_PATH`
- Constructor argument: `CyllamaDB(path="/custom/path.db")`
- Disable entirely: `CyllamaDB(enabled=False)` or `CYLLAMA_DB_ENABLED=0`

## Schema Design

### 1. Configuration Table

Stores key-value configuration with JSON support for complex values.

```sql
CREATE TABLE IF NOT EXISTS config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,           -- JSON-encoded value
    value_type TEXT NOT NULL,      -- 'str', 'int', 'float', 'bool', 'json'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_config_key ON config(key);
```

**Example entries:**
| key | value | value_type |
|-----|-------|------------|
| `default_model` | `"models/llama.gguf"` | str |
| `default_temperature` | `0.7` | float |
| `default_max_tokens` | `512` | int |
| `n_gpu_layers` | `99` | int |

### 2. Response Cache Table

Caches responses with content-based hashing and TTL support.

```sql
CREATE TABLE IF NOT EXISTS cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cache_key TEXT UNIQUE NOT NULL,  -- Hash of (prompt + config + model)
    prompt TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,       -- SHA256 of prompt (for indexing)
    model_path TEXT NOT NULL,
    config_json TEXT NOT NULL,       -- Serialized GenerationConfig
    response_text TEXT NOT NULL,
    response_json TEXT,              -- Full Response.to_json() if stats available
    finish_reason TEXT,
    tokens_generated INTEGER,
    tokens_per_second REAL,
    generation_time_ms REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,            -- NULL = never expires
    hit_count INTEGER DEFAULT 0,
    last_hit_at TIMESTAMP
);

CREATE INDEX idx_cache_key ON cache(cache_key);
CREATE INDEX idx_cache_prompt_hash ON cache(prompt_hash);
CREATE INDEX idx_cache_model ON cache(model_path);
CREATE INDEX idx_cache_expires ON cache(expires_at);
```

**Cache Key Generation:**
```python
def generate_cache_key(prompt: str, model_path: str, config: GenerationConfig) -> str:
    """Generate deterministic cache key from inputs."""
    # Normalize config to dict, excluding non-deterministic fields
    config_dict = {
        'temperature': config.temperature,
        'max_tokens': config.max_tokens,
        'top_k': config.top_k,
        'top_p': config.top_p,
        'min_p': config.min_p,
        'repeat_penalty': config.repeat_penalty,
        'stop_sequences': sorted(config.stop_sequences),
        # Note: seed excluded for deterministic caching
    }

    key_data = json.dumps({
        'prompt': prompt,
        'model': os.path.basename(model_path),  # Normalize path
        'config': config_dict
    }, sort_keys=True)

    return hashlib.sha256(key_data.encode()).hexdigest()
```

### 3. Statistics Table

Tracks per-generation statistics for analytics.

```sql
CREATE TABLE IF NOT EXISTS stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,                 -- Groups generations in same session
    model_path TEXT NOT NULL,
    model_name TEXT,                 -- Extracted from model metadata
    prompt_tokens INTEGER,
    generated_tokens INTEGER,
    total_tokens INTEGER,
    prompt_time_ms REAL,
    generation_time_ms REAL,
    total_time_ms REAL,
    tokens_per_second REAL,
    temperature REAL,
    max_tokens INTEGER,
    finish_reason TEXT,
    cache_hit BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_stats_model ON stats(model_path);
CREATE INDEX idx_stats_session ON stats(session_id);
CREATE INDEX idx_stats_created ON stats(created_at);
```

### 4. Logs Table

Structured logging for debugging and auditing.

```sql
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    level TEXT NOT NULL,             -- DEBUG, INFO, WARNING, ERROR
    category TEXT,                   -- 'generation', 'cache', 'config', 'error'
    message TEXT NOT NULL,
    context_json TEXT,               -- Additional structured context
    model_path TEXT,
    prompt_preview TEXT,             -- First 100 chars of prompt
    error_type TEXT,
    error_message TEXT,
    stack_trace TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_logs_level ON logs(level);
CREATE INDEX idx_logs_category ON logs(category);
CREATE INDEX idx_logs_created ON logs(created_at);
```

### 5. Models Table (Optional)

Cache model metadata for quick access without loading.

```sql
CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    filename TEXT NOT NULL,
    size_bytes INTEGER,
    format TEXT,                     -- 'gguf', etc.
    architecture TEXT,               -- 'llama', 'mistral', etc.
    parameter_count TEXT,            -- '1B', '7B', etc.
    quantization TEXT,               -- 'Q4_0', 'Q8_0', etc.
    context_length INTEGER,
    vocab_size INTEGER,
    chat_template TEXT,              -- Cached template string
    metadata_json TEXT,              -- Full GGUF metadata
    first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMP
);

CREATE INDEX idx_models_path ON models(path);
```

## Python API Design

### CyllamaDB Class

```python
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import sqlite3
import json
import hashlib
import os

@dataclass
class CacheEntry:
    """Cached response entry."""
    cache_key: str
    prompt: str
    model_path: str
    response_text: str
    response_json: Optional[str]
    created_at: datetime
    expires_at: Optional[datetime]
    hit_count: int

class CyllamaDB:
    """SQLite-based storage for cyllama."""

    def __init__(
        self,
        path: Optional[str] = None,
        enabled: bool = True,
        cache_ttl: Optional[timedelta] = timedelta(days=7),
        max_cache_size_mb: float = 100.0,
        log_level: str = "INFO"
    ):
        """
        Initialize the database.

        Args:
            path: Custom database path. None = default location.
            enabled: If False, all operations are no-ops.
            cache_ttl: Default cache entry TTL. None = never expires.
            max_cache_size_mb: Maximum cache size before cleanup.
            log_level: Minimum log level to store.
        """
        ...

    # === Configuration API ===

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        ...

    def set_config(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        ...

    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration as a dictionary."""
        ...

    # === Caching API ===

    def get_cached_response(
        self,
        prompt: str,
        model_path: str,
        config: GenerationConfig
    ) -> Optional[Response]:
        """
        Look up cached response.

        Returns None if not found or expired.
        Updates hit_count and last_hit_at on hit.
        """
        ...

    def cache_response(
        self,
        prompt: str,
        model_path: str,
        config: GenerationConfig,
        response: Response,
        ttl: Optional[timedelta] = None  # None = use default
    ) -> None:
        """Cache a response."""
        ...

    def invalidate_cache(
        self,
        prompt: Optional[str] = None,
        model_path: Optional[str] = None,
        older_than: Optional[datetime] = None
    ) -> int:
        """
        Invalidate cache entries matching criteria.

        Returns number of entries removed.
        """
        ...

    def cleanup_cache(self) -> int:
        """Remove expired entries and enforce size limit."""
        ...

    # === Statistics API ===

    def record_generation(
        self,
        model_path: str,
        response: Response,
        config: GenerationConfig,
        cache_hit: bool = False
    ) -> None:
        """Record generation statistics."""
        ...

    def get_stats_summary(
        self,
        model_path: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get aggregated statistics.

        Returns:
            {
                'total_generations': int,
                'total_tokens': int,
                'avg_tokens_per_second': float,
                'cache_hit_rate': float,
                'by_model': {...},
                'by_day': [...],
            }
        """
        ...

    def get_generation_history(
        self,
        limit: int = 100,
        offset: int = 0,
        model_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get recent generation records."""
        ...

    # === Logging API ===

    def log(
        self,
        level: str,
        message: str,
        category: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None
    ) -> None:
        """Log a message to the database."""
        ...

    def get_logs(
        self,
        level: Optional[str] = None,
        category: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Query logs."""
        ...

    # === Maintenance ===

    def vacuum(self) -> None:
        """Compact the database file."""
        ...

    def get_db_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            {
                'file_size_mb': float,
                'cache_entries': int,
                'cache_size_mb': float,
                'stats_entries': int,
                'log_entries': int,
            }
        """
        ...

    def close(self) -> None:
        """Close database connection."""
        ...
```

## Integration with Existing API

### Option 1: Global Database (Recommended)

```python
from cyllama import complete, get_db, configure_db

# Configure once at startup
configure_db(
    enabled=True,
    cache_ttl=timedelta(hours=24),
    log_level="DEBUG"
)

# All generation functions automatically use the global DB
response = complete("Hello", model_path="model.gguf")  # May be cached

# Access DB directly for queries
db = get_db()
stats = db.get_stats_summary(since=datetime.now() - timedelta(days=7))
print(f"Generated {stats['total_tokens']} tokens this week")
```

### Option 2: Explicit Database

```python
from cyllama import LLM, CyllamaDB

db = CyllamaDB(path="./my_project.db")

llm = LLM("model.gguf", db=db)

# Or with complete()
response = complete("Hello", model_path="model.gguf", db=db)
```

### Option 3: Decorator-Based Caching

```python
from cyllama import LLM, cached

llm = LLM("model.gguf")

# Cache this specific call
response = cached(ttl=timedelta(hours=1))(llm)("Expensive prompt")

# Or as decorator
@cached(ttl=timedelta(days=1))
def generate_summary(text: str) -> str:
    return llm(f"Summarize: {text}")
```

## Configuration Options

### Environment Variables

```bash
# Database location
CYLLAMA_DB_PATH=~/.local/share/cyllama/cyllama.db

# Enable/disable
CYLLAMA_DB_ENABLED=1          # 1=enabled (default), 0=disabled

# Cache settings
CYLLAMA_CACHE_TTL=604800      # Seconds (default: 7 days)
CYLLAMA_CACHE_MAX_SIZE_MB=100 # Maximum cache size

# Logging
CYLLAMA_LOG_LEVEL=INFO        # DEBUG, INFO, WARNING, ERROR
CYLLAMA_LOG_TO_DB=1           # Also log to database
```

### Programmatic Configuration

```python
from cyllama import configure_db

configure_db(
    enabled=True,
    path="~/.cyllama/data.db",
    cache_ttl=timedelta(days=7),
    max_cache_size_mb=100.0,
    log_level="INFO",
    log_to_db=True,
    auto_vacuum=True,  # Run VACUUM periodically
)
```

## Caching Strategy

### Cache Key Components

1. **Prompt** - Full prompt text (hashed)
2. **Model** - Model filename (not full path for portability)
3. **Deterministic Config** - temperature, max_tokens, top_k, top_p, min_p, repeat_penalty, stop_sequences

### Excluded from Cache Key

- `seed` - Different seeds should return cached result
- `n_gpu_layers` - Infrastructure detail, doesn't affect output
- `n_ctx` - Context size doesn't affect generation content
- `verbose` - Logging flag

### Cache Invalidation

```python
# Invalidate all cache
db.invalidate_cache()

# Invalidate for specific model
db.invalidate_cache(model_path="model.gguf")

# Invalidate entries older than 1 day
db.invalidate_cache(older_than=datetime.now() - timedelta(days=1))

# Automatic cleanup (runs periodically)
db.cleanup_cache()  # Removes expired + enforces size limit
```

## Statistics Queries

### Example Analytics

```python
# Get weekly summary
stats = db.get_stats_summary(since=datetime.now() - timedelta(days=7))

print(f"""
Weekly Stats:
  Total generations: {stats['total_generations']}
  Total tokens: {stats['total_tokens']}
  Avg speed: {stats['avg_tokens_per_second']:.1f} tokens/sec
  Cache hit rate: {stats['cache_hit_rate']*100:.1f}%
""")

# Per-model breakdown
for model, model_stats in stats['by_model'].items():
    print(f"  {model}: {model_stats['generations']} generations")

# Export to pandas for analysis
import pandas as pd
history = db.get_generation_history(limit=1000)
df = pd.DataFrame(history)
df.groupby('model_name')['tokens_per_second'].mean()
```

## Thread Safety

- Uses WAL (Write-Ahead Logging) mode for better concurrency
- Connection per-thread with `check_same_thread=False`
- Write operations use transactions
- Read operations don't block writes

```python
# Thread-safe initialization
db = CyllamaDB()  # Configures WAL mode automatically

# Safe to use from multiple threads
from concurrent.futures import ThreadPoolExecutor

def generate(prompt):
    return complete(prompt, model_path="model.gguf")

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(generate, prompts))
```

## Migration Strategy

Schema versioning with automatic migrations:

```sql
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

```python
MIGRATIONS = [
    # Version 1: Initial schema
    """
    CREATE TABLE IF NOT EXISTS config (...);
    CREATE TABLE IF NOT EXISTS cache (...);
    ...
    """,
    # Version 2: Add models table
    """
    CREATE TABLE IF NOT EXISTS models (...);
    """,
    # Version 3: Add index
    """
    CREATE INDEX IF NOT EXISTS idx_cache_created ON cache(created_at);
    """,
]
```

## File Structure

```
src/cyllama/
    db/
        __init__.py         # Exports CyllamaDB, configure_db, get_db
        database.py         # Main CyllamaDB class
        schema.py           # SQL schema and migrations
        cache.py            # Caching logic and key generation
        stats.py            # Statistics aggregation
        logging.py          # Database logging handler
```

## Implementation Phases

### Phase 1: Core Infrastructure
- [ ] `CyllamaDB` class with connection management
- [ ] Schema creation and migrations
- [ ] Basic configuration get/set

### Phase 2: Response Caching
- [ ] Cache key generation
- [ ] `get_cached_response()` / `cache_response()`
- [ ] TTL and size-based cleanup
- [ ] Integration with `complete()` and `LLM()`

### Phase 3: Statistics
- [ ] `record_generation()`
- [ ] `get_stats_summary()` with aggregations
- [ ] `get_generation_history()`

### Phase 4: Logging
- [ ] Database logging handler
- [ ] Log query API
- [ ] Integration with Python logging

### Phase 5: Polish
- [ ] CLI commands for DB management
- [ ] Export/import functionality
- [ ] Performance optimization

## Open Questions

1. **Should caching be on by default?**
   - Pro: Immediate performance benefit
   - Con: Unexpected behavior if prompts have side effects

2. **How to handle model path normalization?**
   - Same model at different paths should share cache?
   - Use model hash from GGUF metadata?

3. **Should we support multiple databases?**
   - Per-project databases for isolation?
   - Global + project overlay?

4. **Integration with async API?**
   - aiosqlite for true async?
   - Or thread pool for DB operations?

## References

- [SQLite WAL Mode](https://www.sqlite.org/wal.html)
- [Python sqlite3 Documentation](https://docs.python.org/3/library/sqlite3.html)
- [XDG Base Directory Specification](https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html)
