# ONNX Integration Layer -- Design Proposal

Status: sketch / parked until concrete model and data format exist

## Motivation

Lab test pipeline: physical samples are tested, results are classified/interpreted by an ONNX model, then an LLM performs trend analysis and writes reports from the classified outputs.

```
ONNX model (classify) --> structured results --> LLM (analyze/report)
```

The integration layer's job is narrow: run an ONNX model, capture its outputs, and format them as LLM input.

## Proposed Location

`cyllama/integrations/onnx.py` -- thin Python wrapper, no Cython or C++ bundling. `onnxruntime` is an optional dependency, imported only when the integration is used (same pattern as the LangChain integration).

## Sketch API

```python
from cyllama.integrations.onnx import ONNXModel
from cyllama import LLM, complete

# Stage 1: classify
model = ONNXModel("lab_classifier.onnx")
results = model.predict(sample_data)

# Stage 2: analyze + report
response = complete(
    f"Analyze these lab results and write a report:\n{results.to_prompt()}",
    model_path="model.gguf"
)
```

## Key Design Decisions

1. **How much to wrap** -- thin wrapper around `onnxruntime.InferenceSession` with a consistent API, or just a helper that handles the ONNX-to-prompt formatting. Leaning toward thin wrapper plus formatting helpers.

2. **Data handoff format** -- how structured ONNX outputs (class labels, probabilities, feature vectors) get serialized into LLM-consumable text. Needs to be driven by actual data shapes from the lab classifier.

3. **Optional dependency** -- `onnxruntime` only imported when `cyllama.integrations.onnx` is used. Import error should give a clear message pointing to `pip install onnxruntime`.

## What To Defer

- Do not build until a concrete ONNX model and input data format exist to design against.

- The risk of building speculatively is designing an API that does not fit the actual data shapes when they materialize.

- The integration itself is small enough to implement quickly once requirements are concrete.

## Open Questions

- What are the input data shapes? (tabular, time-series, spectral, images of samples?)

- Single classification model or an ensemble/pipeline of ONNX models?

- Does the LLM stage need raw probabilities or just top-k class labels?

- Batch processing (many samples -> one report) vs. single-sample workflows?
