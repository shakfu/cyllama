"""Tests for the mtmd batch-encoding and video-input bindings.

Both need a real vision model plus its mmproj, so they skip cleanly when
those are absent. The video tests additionally need ffmpeg/ffprobe on PATH,
and synthesise their own clip rather than shipping a binary fixture.
"""

import gc
import shutil
import subprocess
from pathlib import Path

import pytest

import cyllama.llama.llama_cpp as cy

ROOT = Path.cwd()
VISION_MODEL = ROOT / "models" / "gemma-4-E4B-it-Q4_K_M.gguf"
MMPROJ = ROOT / "models" / "mmproj-gemma-4-E4B-it-BF16.gguf"

HAS_VISION = VISION_MODEL.exists() and MMPROJ.exists()
HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

pytestmark = pytest.mark.skipif(not HAS_VISION, reason="vision model / mmproj not available")


@pytest.fixture(scope="module")
def vision():
    """Loaded vision model plus its mtmd context."""
    params = cy.LlamaModelParams()
    params.n_gpu_layers = -1
    model = cy.LlamaModel(str(VISION_MODEL), params)
    mctx = cy.MtmdContext(str(MMPROJ), model, cy.MtmdContextParams())
    yield model, mctx
    mctx.close()
    model.close()
    gc.collect()


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """A short synthetic test video."""
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg/ffprobe not on PATH")
    path = tmp_path_factory.mktemp("video") / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=3:size=320x240:rate=10",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


class TestMmprojCaps:
    def test_reports_modalities_without_loading_context(self):
        caps = cy.get_mmproj_caps(str(MMPROJ))
        assert set(caps) == {"vision", "audio"}
        assert caps["vision"] is True

    def test_agrees_with_loaded_context(self, vision):
        _, mctx = vision
        caps = cy.get_mmproj_caps(str(MMPROJ))
        assert caps["vision"] == mctx.supports_vision
        assert caps["audio"] == mctx.supports_audio

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            cy.get_mmproj_caps("not-a-real-mmproj.gguf")


class TestModelCanChat:
    def test_returns_bool(self, vision):
        model, mctx = vision
        ctx_params = cy.LlamaContextParams()
        ctx_params.n_ctx = 2048
        lctx = cy.LlamaContext(model, ctx_params)
        try:
            assert mctx.model_can_chat(lctx) is True
        finally:
            lctx.close()
            gc.collect()


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")
class TestMtmdVideo:
    def test_info_matches_source(self, vision, clip):
        _, mctx = vision
        with mctx.open_video(str(clip), fps_target=1.0) as video:
            info = video.info
        assert info["width"] == 320
        assert info["height"] == 240
        assert info["fps"] == pytest.approx(1.0)

    def test_iteration_yields_frames_and_text(self, vision, clip):
        _, mctx = vision
        kinds = []
        with mctx.open_video(str(clip), fps_target=1.0) as video:
            for kind, value in video:
                kinds.append(kind)
                if kind == "image":
                    assert isinstance(value, cy.MtmdBitmap)
                    assert value.width == 320
                    assert value.height == 240
                else:
                    assert isinstance(value, str)
        # 3 seconds at 1 fps.
        assert kinds.count("image") == 3
        assert "text" in kinds

    def test_read_next_returns_none_at_eof(self, vision, clip):
        _, mctx = vision
        with mctx.open_video(str(clip), fps_target=1.0) as video:
            while video.read_next() is not None:
                pass
            assert video.read_next() is None

    def test_fps_target_controls_frame_count(self, vision, clip):
        _, mctx = vision

        def count(fps):
            with mctx.open_video(str(clip), fps_target=fps) as video:
                return sum(1 for kind, _ in video if kind == "image")

        assert count(2.0) > count(1.0)

    def test_use_after_close_raises(self, vision, clip):
        _, mctx = vision
        video = mctx.open_video(str(clip), fps_target=1.0)
        video.close()
        with pytest.raises(RuntimeError):
            video.read_next()

    def test_missing_file_raises(self, vision):
        _, mctx = vision
        with pytest.raises(FileNotFoundError):
            mctx.open_video("not-a-real-video.mp4")


class TestMtmdBatch:
    @staticmethod
    def _media_chunks(mctx, bitmaps):
        chunks = mctx.tokenize(mctx.marker * len(bitmaps), bitmaps, True, True)
        media = [chunks[i] for i in range(len(chunks)) if chunks[i].type != cy.MtmdInputChunkType.TEXT]
        # Keep `chunks` alive: the batch borrows the chunk pointers.
        return chunks, media

    @pytest.fixture
    def bitmaps(self, vision, clip):
        _, mctx = vision
        frames = []
        with mctx.open_video(str(clip), fps_target=1.0) as video:
            for kind, value in video:
                if kind == "image":
                    frames.append(value)
        return frames[:2]

    @pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")
    def test_encode_and_read_embeddings(self, vision, bitmaps):
        model, mctx = vision
        chunks, media = self._media_chunks(mctx, bitmaps)
        assert len(media) == 2

        with mctx.batch() as batch:
            for chunk in media:
                batch.add_chunk(chunk)
            batch.encode()

            embd = batch.get_output_embd(media[0], model.n_embd)
            assert len(embd) == media[0].n_tokens
            assert all(len(row) == model.n_embd for row in embd)
        assert chunks is not None

    @pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")
    def test_rejects_text_chunks(self, vision, bitmaps):
        _, mctx = vision
        chunks, _ = self._media_chunks(mctx, bitmaps)
        text_chunks = [chunks[i] for i in range(len(chunks)) if chunks[i].type == cy.MtmdInputChunkType.TEXT]
        assert text_chunks, "expected the prompt to produce at least one text chunk"

        with mctx.batch() as batch:
            with pytest.raises(ValueError):
                batch.add_chunk(text_chunks[0])

    @pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")
    def test_rejects_bad_n_embd(self, vision, bitmaps):
        _, mctx = vision
        _, media = self._media_chunks(mctx, bitmaps)
        with mctx.batch() as batch:
            batch.add_chunk(media[0])
            batch.encode()
            with pytest.raises(ValueError):
                batch.get_output_embd(media[0], 0)

    def test_use_after_close_raises(self, vision):
        _, mctx = vision
        batch = mctx.batch()
        batch.close()
        with pytest.raises(RuntimeError):
            batch.encode()

    def test_close_is_idempotent(self, vision):
        _, mctx = vision
        batch = mctx.batch()
        batch.close()
        batch.close()
