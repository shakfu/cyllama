# fixes

## token_to_piece fix

Problem: The `token_to_piece` method in `src/cyllama/llama_cpp.pyx` was
producing corrupted text output with replacement characters (�) because it
 was reading uninitialized memory beyond the actual string content.

Root Cause: The `llama_token_to_piece` function doesn't null-terminate the
buffer, but the Cython code was calling .decode() on the entire 128-byte
buffer instead of just the bytes that were actually written.

Solution: Modified the `token_to_piece` method to:
1. Capture the return value from `llama_token_to_piece`, which indicates the
 number of bytes written
2. Use buffer slicing `buf[:length]` to only decode the actual content
3. Add error handling for negative return values

Result: Clean, corruption-free text output. The test now produces:
```
When did the universe begin?

response: The age of the universe is estimated to be around 13.8 billion
years. The universe began as a singularity, a point of infinite density
and zero
```
The fix is minimal, safe, and follows the llama.cpp API correctly by using
 the returned length value as intended.

