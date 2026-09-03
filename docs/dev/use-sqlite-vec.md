# Replacing sqlite-vector with sqlite-vec

Feasibility study for swapping the vendored
[`sqlite-vector`](https://github.com/sqliteai/sqlite-vector) extension
behind `SqliteVectorStore` for
[`sqlite-vec`](https://github.com/asg017/sqlite-vec).

Status: **investigation complete, swap not performed.** A `SqliteVecStore`
adapter shipped instead (`src/cyllama/rag/stores/sqlite_vec.py`), so users
can opt into sqlite-vec today without cyllama committing to it as the
default backend. This document records why, and what a full swap would
still cost.

Measurements below were taken on 2026-09-03 against sqlite-vector 1.0.0
(the vendored copy in `thirdparty/sqlite-vector`), sqlite-vec v0.1.9 and
sqlite-vec v0.1.10-alpha.4, on Linux x86_64 / CPython 3.13.

## Motivation: the license

| | sqlite-vector 1.0.0 | sqlite-vec |
| --- | --- | --- |
| License | Elastic License 2.0, modified | Apache-2.0 **OR** MIT |
| Open-source use | Free, but only inside OSI-licensed open-source projects | Free |
| Commercial / production use | Requires a paid license from SQLite Cloud, Inc. | Free |
| Hosted-service use | Prohibited without explicit licensing | Free |

cyllama itself is MIT, so vendoring sqlite-vector is permitted. The
problem is downstream: anyone who builds a commercial product on
cyllama's RAG stack inherits an obligation to buy a license from SQLite
Cloud, and nothing in the install flow tells them so. sqlite-vec's
Apache-2.0/MIT dual license removes that entirely.

## Candidate assessment

sqlite-vec as of this study: ~8.1k stars, last stable **v0.1.9**
(2026-03-31), alpha **v0.1.10-alpha.4** (2026-05-18), 204 open issues,
last commit 2026-05-18. Sponsored by Mozilla Builders plus Fly.io,
Turso, SQLite Cloud and Shinkai. The README still carries an explicit
"pre-v1, so expect breaking changes" warning.

That is the main counterweight to the license win: swapping the default
backend would trade a 1.0.0 release for a pre-1.0 one whose most recent
release is an alpha.

## Architectural difference

sqlite-vector attaches to an *existing ordinary table*: you declare a
`BLOB` column and call `vector_init('table', 'column', ...)`. sqlite-vec
instead owns a `vec0` **virtual table**.

That matters because **SQLite forbids triggers on virtual tables**:

```
sqlite> CREATE TRIGGER t AFTER INSERT ON some_vec0_table BEGIN SELECT 1; END;
Error: cannot create triggers on virtual tables
```

`HybridStore` (`src/cyllama/rag/advanced.py`) keeps its FTS5 index in
sync with three `AFTER INSERT/UPDATE/DELETE` triggers on the embeddings
table. Moving chunk rows into `vec0` would break that outright.

The fix is to keep the base table plain and hang a `vec0` sidecar off
it, keyed by the base table's `id`:

```sql
CREATE TABLE embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    metadata TEXT
);
CREATE VIRTUAL TABLE embeddings_vec USING vec0(
    embedding float[384] distance_metric=cosine
);
```

with search becoming a join back onto the base table:

```sql
SELECT e.id, e.text, e.metadata, v.distance
FROM embeddings AS e
JOIN (SELECT rowid, distance FROM embeddings_vec
      WHERE embedding MATCH ? AND k = ?) AS v ON e.id = v.rowid
ORDER BY v.distance IS NULL, v.distance;
```

This preserves the metadata table, the source-dedup table, the FTS5
triggers and the overall shape of `search()`. It is also the design the
shipped `SqliteVecStore` adapter uses.

## Feasibility: verified against the real test suite

A full port of `src/cyllama/rag/store.py` to sqlite-vec was written as a
**136-line diff** and run against cyllama's own RAG tests
(`test_rag_store.py`, `test_rag_advanced.py`, `test_rag_dedup.py`) with
`SqliteVectorStore` monkeypatched to the port:

| | v0.1.9 | v0.1.10-alpha.4 |
| --- | --- | --- |
| Passed | 127 | 127 |
| Failed | 3 | 3 |
| Skipped | 4 (reranker model unavailable) | 4 |

All of `TestHybridStore` and the entire dedup suite passed, confirming
the sidecar design keeps FTS5 working.

The 3 failures are the same on both versions and are genuine capability
gaps, not porting bugs:

* `test_dot_metric` and `test_init_all_metrics` — `vec0` offers cosine,
  L2 and L1. There is no dot-product metric and no `vec_distance_dot()`.
* `test_init_all_vector_types` — `vec0` offers float32, int8 and bit.
  There is no `uint8`.

Neither is used anywhere outside the tests; the defaults are cosine and
float32, and nothing in `pipeline.py` or `rag.py` sets either.

## Behavioural differences found

Four things bit during the port and are worth recording:

1. **NULL cosine distance for zero-norm vectors.** `vec0` reports SQL
   `NULL` for the cosine distance to a zero vector (mathematically
   undefined); sqlite-vector returned a number. Unhandled, this raises
   `TypeError: unsupported operand type(s) for -: 'float' and 'NoneType'`
   mid-iteration, which in the concurrency test left an unfinalized
   statement and produced a cascade of spurious `database is locked`
   errors in the other threads. It looks like a locking bug and isn't
   one. Handle NULL explicitly and sort it last
   (`ORDER BY v.distance IS NULL, v.distance` — SQLite sorts NULL first
   by default).

2. **Blob element type must match the column.** sqlite-vector accepted a
   float32 blob for every column type and converted internally. `vec0`
   reads a bare `BLOB` as float32 always, so an int8 vector must be
   tagged with `vec_int8(?)` on **both** insert and query, or you get
   `Inserted vector for the "embedding" column is expected to be of type
   int8, but a float32 vector was provided`.

3. **`float16` is declared but not implemented.** `vec0` parses a
   `float16[N]` column declaration without complaint, but as of v0.1.9
   it stores those columns as float32 — `vec_type()` reports `float32`
   and each element still occupies 4 bytes:

   ```
   float   : stored=32B vec_type=float32 len=8
   float16 : stored=32B vec_type=float32 len=8
   ```

   Offering `vector_type="float16"` would promise a halving that doesn't
   happen, so the shipped adapter rejects it.

4. **No squared-L2 metric.** Request L2 and square the distance in
   Python; squaring is monotonic, so the ordering is identical.

## Performance

20,000 vectors × 384 dimensions, cosine, k=10, 200 queries, single
thread. Both extensions built from source with `-O3` and no explicit
SIMD flags.

| | sqlite-vector 1.0.0 | sqlite-vec 0.1.9 |
| --- | --- | --- |
| Exact search | 13.04 ms/query | **7.73 ms/query** |
| Insert 20k | 0.03 s | 0.12 s |
| Database size | 41.1 MB | **32.3 MB** |
| Top-10 agreement | — | 100% |

sqlite-vec is ~1.7× faster on exhaustive search and produces a smaller
database, with identical results.

### The one real regression: quantization

`SqliteVectorStore` exposes `quantize()` and `preload_quantization()`,
backed by sqlite-vector's `vector_quantize()` / `vector_quantize_scan()`.
sqlite-vec 0.1.9 has no equivalent: a `vec0` table is either exhaustive
or built with an ANN index at `CREATE TABLE` time. There is no runtime
quantization step to call.

Same corpus as above, measuring recall@10 against exact search:

| Approach | Latency | Recall@10 |
| --- | --- | --- |
| sqlite-vector `vector_quantize_scan` | 1.85 ms/q | 96.2% |
| sqlite-vec 0.1.9, binary quantize + 8× rescore | 1.00 ms/q | 30.7% |
| sqlite-vec 0.1.10-alpha.4, `indexed by diskann(neighbor_quantizer=binary)` | 0.74 ms/q | 32.4% |
| sqlite-vec 0.1.10-alpha.4, `indexed by diskann(neighbor_quantizer=int8)` | 3.48 ms/q | 70.6% |

**Read these recall numbers with care.** The corpus is random Gaussian
vectors, which is the adversarial worst case for graph-based ANN: there
is no cluster structure to exploit and all pairwise distances are nearly
equal. Real embeddings cluster, and DiskANN would score far better on
them. The comparison is still informative in one specific way:
sqlite-vector's approach is an *exhaustive scan over quantized vectors*,
so its recall is insensitive to how the data is distributed, whereas
sqlite-vec's is a graph traversal and is not. A benchmark on real
embeddings is needed before treating the DiskANN row as a verdict.

Note also that the alpha's DiskANN index took 53 s (binary) and 131 s
(int8) to build for 20k vectors inserted row by row, versus 0.1 s for a
flat table.

## Cost of a full swap

Vendoring gets *simpler*: sqlite-vec is a single 320 KB `sqlite-vec.c`
amalgamation plus a generated header, against sqlite-vector's eight
source files. It compiles warning-clean with the project's existing
flags, needs only `sqlite3ext.h`, has explicit `_WIN32` / `intrin.h`
handling and guarded NEON/AVX paths, and produces a 165 KB `.so` versus
sqlite-vector's 191 KB.

| Area | Change |
| --- | --- |
| `thirdparty/` | Swap 8 files for 1 |
| `CMakeLists.txt` | ~10 lines (the `vector` target's source list) |
| `scripts/manage.py` | ~30 lines (`SqliteVectorBuilder`, version constant) |
| `src/cyllama/rag/store.py` | ~136 lines |
| `tests/` | 3 tests (`dot` metric, `uint8` type) |
| Docs / README / CHANGELOG | Prose |
| **CI workflows** | **None** — see below |
| On-disk format | Breaking; existing `.db` stores need a rebuild |

CI needs no changes as long as the built artifact keeps the name
`vector.{so,dylib,dll}`: both `build-cibw-abi3.yml` and
`build-gpu-wheels-abi3.yml` skip the ABI3 check via
`p.stem.split('.')[0] == 'vector'`. sqlite-vec's init symbol is
`sqlite3_vec_init`, which won't match a file named `vector.so`, so load
it with an explicit entrypoint:

```python
conn.load_extension(path, entrypoint="sqlite3_vec_init")
```

The `entrypoint` keyword arrived in Python 3.12; cyllama already
requires 3.12. (mypy's stdlib stubs don't carry it yet, so the call
needs `# type: ignore[call-arg]`.)

## Recommendation

**Ship the adapter now; revisit the default later.** That is what was
done.

Shipping `SqliteVecStore` in `cyllama.rag.stores` gives commercial users
a permissively-licensed path today at zero risk to the default backend,
and gives the project real usage data before betting the default on a
pre-v1 dependency.

Reconsider swapping the default when either of these holds:

* sqlite-vec reaches v1.0, or the DiskANN/IVF work in the 0.1.10 line
  lands in a stable release with recall verified on real embeddings; or
* the licensing situation becomes an active problem for users — at which
  point the swap is a known ~136-line change with a green test suite
  behind it.

If the swap does happen, `quantize()` / `preload_quantization()` are the
open design question. Options, roughly in order of preference:

1. Deprecate them to no-ops and re-express the choice as a create-time
   index selection (`SqliteVecStore(..., index="diskann")`), matching
   how `vec0` actually works.
2. Keep them raising `NotImplementedError` on the sqlite-vec backend and
   leave them working only on the sqlite-vector one during a transition.
3. Emulate with `vec_quantize_binary()` plus a rescore pass — measured
   above at 30.7% recall on this corpus, so not viable without real-
   embedding numbers that say otherwise.

## Using sqlite-vec today

```bash
pip install sqlite-vec
```

```python
from cyllama.rag import RAG
from cyllama.rag.stores import SqliteVecStore

store = SqliteVecStore(dimension=384, db_path="vectors.db")
rag = RAG(embedding_model="models/bge-small.gguf",
          generation_model="models/llama.gguf",
          store=store)
```

See `src/cyllama/rag/stores/sqlite_vec.py` and
`tests/test_rag_sqlite_vec.py`. The adapter takes an `extension_path=`
argument for pointing at your own build of the extension instead of the
copy shipped by the PyPI package.

The adapter is covered in CI by the `test-store-adapters` workflow,
which runs its test module and the cross-backend conformance suite
(`tests/test_rag_store_conformance.py`) against the current sqlite-vec
release on a weekly cron. That is also the tripwire for the pre-v1
concern above: a breaking change upstream shows up there within a week
rather than in a user's bug report.
