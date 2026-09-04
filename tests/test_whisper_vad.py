"""Tests for the standalone whisper VAD API, parallel transcription and language ID.

The VAD tests need a Silero VAD model converted to ggml. Download with:

    curl -L -o models/ggml-silero-v5.1.2.bin \\
      https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin

They skip cleanly when it is absent.
"""

import gc
import wave
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

import cyllama.whisper.whisper_cpp as wcy

# Anchored to this file, not the cwd: see the note in conftest.py.
ROOT = Path(__file__).resolve().parent.parent
WHISPER_MODEL = ROOT / "models" / "ggml-base.en.bin"
VAD_MODEL = ROOT / "models" / "ggml-silero-v5.1.2.bin"
JFK_WAV = ROOT / "tests" / "samples" / "jfk.wav"

SAMPLE_RATE = 16000


def load_pcm(path):
    """Read a 16-bit mono wav as float32 in [-1, 1]."""
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SAMPLE_RATE
        assert w.getsampwidth() == 2
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


@pytest.fixture(scope="module")
def pcm():
    if not JFK_WAV.exists():
        pytest.skip("tests/samples/jfk.wav not available")
    return load_pcm(JFK_WAV)


@pytest.mark.skipif(not VAD_MODEL.exists(), reason="Silero VAD model not available")
class TestWhisperVadContext:
    def test_segments_from_samples(self, pcm):
        with wcy.WhisperVadContext(str(VAD_MODEL)) as vad:
            with vad.segments_from_samples(pcm) as segments:
                assert len(segments) > 0
                duration_cs = len(pcm) / SAMPLE_RATE * 100
                for t0, t1 in segments:
                    # Timestamps are in 10 ms units, ordered, and inside the clip.
                    assert 0 <= t0 < t1 <= duration_cs

    def test_detect_speech_then_segments_from_probs(self, pcm):
        with wcy.WhisperVadContext(str(VAD_MODEL)) as vad:
            assert vad.detect_speech(pcm) is True
            n = vad.n_probs()
            assert n > 0

            probs = vad.probs()
            assert len(probs) == n
            assert all(0.0 <= p <= 1.0 for p in probs)
            # Speech is present, so at least one frame must be confident.
            assert max(probs) > 0.5

            with vad.segments_from_probs() as segments:
                assert len(segments) > 0

    def test_both_segment_paths_agree(self, pcm):
        """segments_from_samples is detect_speech + segments_from_probs natively."""
        with wcy.WhisperVadContext(str(VAD_MODEL)) as vad:
            with vad.segments_from_samples(pcm) as a:
                from_samples = list(a)
            vad.detect_speech(pcm)
            with vad.segments_from_probs() as b:
                from_probs = list(b)
        assert from_samples == from_probs

    def test_indexing_and_iteration(self, pcm):
        with wcy.WhisperVadContext(str(VAD_MODEL)) as vad:
            with vad.segments_from_samples(pcm) as segments:
                assert segments[0] == list(segments)[0]
                assert segments[-1] == list(segments)[len(segments) - 1]
                with pytest.raises(IndexError):
                    segments[len(segments)]

    def test_custom_params_change_segmentation(self, pcm):
        """A high threshold should not detect more speech than a low one."""
        with wcy.WhisperVadContext(str(VAD_MODEL)) as vad:
            loose = wcy.WhisperVadParams()
            loose.threshold = 0.2
            strict = wcy.WhisperVadParams()
            strict.threshold = 0.95

            with vad.segments_from_samples(pcm, loose) as a:
                loose_span = sum(t1 - t0 for t0, t1 in a)
            with vad.segments_from_samples(pcm, strict) as b:
                strict_span = sum(t1 - t0 for t0, t1 in b)

        assert strict_span <= loose_span

    def test_reset_state(self, pcm):
        with wcy.WhisperVadContext(str(VAD_MODEL)) as vad:
            vad.detect_speech(pcm[: SAMPLE_RATE * 2], reset_state=False)
            vad.reset_state()
            # Still usable afterwards.
            assert vad.detect_speech(pcm) is True

    def test_use_after_close_raises(self, pcm):
        vad = wcy.WhisperVadContext(str(VAD_MODEL))
        vad.close()
        with pytest.raises(RuntimeError):
            vad.detect_speech(pcm)

    def test_close_is_idempotent(self):
        vad = wcy.WhisperVadContext(str(VAD_MODEL))
        vad.close()
        vad.close()

    def test_rejects_empty_samples(self):
        with wcy.WhisperVadContext(str(VAD_MODEL)) as vad:
            with pytest.raises(ValueError):
                vad.detect_speech(np.zeros(0, dtype=np.float32))

    def test_rejects_wrong_dtype(self):
        with wcy.WhisperVadContext(str(VAD_MODEL)) as vad:
            with pytest.raises(TypeError):
                vad.detect_speech(np.zeros(1600, dtype=np.float64))


class TestVadContextParams:
    def test_defaults_and_setters(self):
        params = wcy.WhisperVadContextParams()
        assert isinstance(params.n_threads, int)
        params.n_threads = 2
        assert params.n_threads == 2
        params.use_gpu = False
        assert params.use_gpu is False
        params.gpu_device = 0
        assert params.gpu_device == 0


def test_bad_vad_model_path_raises():
    with pytest.raises((FileNotFoundError, ValueError)):
        wcy.WhisperVadContext("definitely-not-a-real-vad-model.bin")


@pytest.mark.skipif(not WHISPER_MODEL.exists(), reason="whisper model not available")
class TestFullParallelAndLangDetect:
    def test_full_parallel_produces_segments(self, pcm):
        ctx = wcy.WhisperContext(str(WHISPER_MODEL))
        try:
            params = wcy.WhisperFullParams()
            params.n_threads = 4
            params.print_progress = False
            ctx.full_parallel(pcm, params, 2)

            assert ctx.full_n_segments() > 0
            text = " ".join(ctx.full_get_segment_text(i) for i in range(ctx.full_n_segments())).lower()
            # Accuracy degrades at chunk boundaries, but content words survive.
            assert "country" in text
        finally:
            ctx.close()
            gc.collect()

    def test_full_parallel_rejects_bad_processor_count(self, pcm):
        ctx = wcy.WhisperContext(str(WHISPER_MODEL))
        try:
            with pytest.raises(ValueError):
                ctx.full_parallel(pcm, None, 0)
        finally:
            ctx.close()
            gc.collect()

    def test_lang_auto_detect_rejects_english_only_model(self, pcm):
        """base.en has no language tokens; detection would return noise."""
        ctx = wcy.WhisperContext(str(WHISPER_MODEL))
        try:
            ctx.pcm_to_mel(pcm, 4)
            with pytest.raises(RuntimeError, match="multilingual"):
                ctx.lang_auto_detect(0, 4)
        finally:
            ctx.close()
            gc.collect()

    def test_pcm_to_mel_then_encode(self, pcm):
        ctx = wcy.WhisperContext(str(WHISPER_MODEL))
        try:
            ctx.pcm_to_mel(pcm, 4)
            ctx.encode(0, 4)
        finally:
            ctx.close()
            gc.collect()

    def test_pcm_to_mel_rejects_empty(self):
        ctx = wcy.WhisperContext(str(WHISPER_MODEL))
        try:
            with pytest.raises(ValueError):
                ctx.pcm_to_mel(np.zeros(0, dtype=np.float32), 4)
        finally:
            ctx.close()
            gc.collect()


@pytest.mark.skipif(
    not (WHISPER_MODEL.exists() and VAD_MODEL.exists()),
    reason="whisper and/or VAD model not available",
)
class TestFullVadSegments:
    def test_internal_vad_segments_readback(self, pcm):
        """params.vad=True runs VAD inside full(); the spans it used are
        now readable, and must match the standalone VAD on the same audio."""
        ctx = wcy.WhisperContext(str(WHISPER_MODEL))
        try:
            params = wcy.WhisperFullParams()
            params.n_threads = 4
            params.print_progress = False
            params.vad = True
            params.vad_model_path = str(VAD_MODEL)
            ctx.full(pcm, params)

            n = ctx.full_n_vad_segments()
            assert n > 0
            internal = [(ctx.full_get_vad_segment_t0(i), ctx.full_get_vad_segment_t1(i)) for i in range(n)]
            for t0, t1 in internal:
                assert 0 <= t0 < t1

            with wcy.WhisperVadContext(str(VAD_MODEL)) as vad:
                with vad.segments_from_samples(pcm) as segments:
                    standalone = [(int(t0), int(t1)) for t0, t1 in segments]

            assert [(int(a), int(b)) for a, b in internal] == standalone
        finally:
            ctx.close()
            gc.collect()

    def test_vad_segment_index_out_of_range(self, pcm):
        ctx = wcy.WhisperContext(str(WHISPER_MODEL))
        try:
            with pytest.raises(IndexError):
                ctx.full_get_vad_segment_t0(0)
        finally:
            ctx.close()
            gc.collect()
