"""Tests for the mtmd batch-encoding and video-input bindings.

Both need a real vision model plus its mmproj, so they skip cleanly when
those are absent. The video tests additionally need ffmpeg/ffprobe on PATH,
and synthesise their own clip rather than shipping a binary fixture.
"""

import gc
import shutil
import subprocess
import sys

import pytest

import cyllama.llama.llama_cpp as cy

from conftest import GEMMA4_MMPROJ as MMPROJ
from conftest import GEMMA4_MODEL as VISION_MODEL

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
        assert caps == {"vision": True, "audio": caps["audio"], "gen_audio": False}

    def test_agrees_with_loaded_context(self, vision):
        _, mctx = vision
        caps = cy.get_mmproj_caps(str(MMPROJ))
        assert caps["vision"] == mctx.supports_vision
        assert caps["audio"] == mctx.supports_audio

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            cy.get_mmproj_caps("not-a-real-mmproj.gguf")


class TestCapabilities:
    def test_vision_model(self, vision):
        _, mctx = vision
        caps = mctx.capabilities
        assert set(caps) == {
            "chat",
            "vision",
            "audio",
            "video",
            "audio_stream",
            "text_gen",
            "audio_gen",
            "audio_gen_spk_ref",
        }
        assert all(type(v) is bool for v in caps.values())
        assert caps["vision"] is True and caps["chat"] is True and caps["text_gen"] is True
        assert caps["audio_gen"] is False and caps["audio_gen_spk_ref"] is False
        assert caps["audio"] == mctx.supports_audio
        # video needs vision, plus the MTMD_VIDEO build option
        assert caps["video"] in (False, caps["vision"])


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

    @staticmethod
    def _items(video):
        with video:
            return [(kind, value.data if kind != "text" else value) for kind, value in video]

    def test_buffer_matches_file(self, vision, clip):
        _, mctx = vision
        from_file = self._items(mctx.open_video(str(clip), fps_target=1.0))
        from_buf = self._items(mctx.open_video_buffer(clip.read_bytes(), fps_target=1.0))
        assert [k for k, _ in from_buf].count("image") == 3
        assert from_buf == from_file

    def test_buffer_rejects_empty_and_garbage(self, vision):
        _, mctx = vision
        with pytest.raises(ValueError):
            mctx.open_video_buffer(b"")
        with pytest.raises(RuntimeError, match="Failed to open video"):
            mctx.open_video_buffer(b"not a video" * 100)

    def test_no_soundtrack_reads_frames_only(self, vision, clip):
        # the clip is silent and gemma-4 does not hear soundtracks, so this checks the
        # option is accepted and changes nothing for frames; the audio path needs LFM3-Omni
        _, mctx = vision
        items = self._items(mctx.open_video(str(clip), fps_target=1.0, no_soundtrack=True))
        assert "audio" not in [k for k, _ in items]
        assert items == self._items(mctx.open_video(str(clip), fps_target=1.0))

    def test_read_after_mtmd_close_raises(self, vision, clip):
        model, _ = vision
        # its own context: closing the module fixture's would break later tests
        mctx = cy.MtmdContext(str(MMPROJ), model, cy.MtmdContextParams())
        video = mctx.open_video(str(clip), fps_target=1.0)
        mctx.close()
        with pytest.raises(RuntimeError, match="MtmdContext is closed"):
            video.read_next()
        with pytest.raises(RuntimeError, match="MtmdContext is closed"):
            video.info
        video.close()
        del video, mctx
        gc.collect()


class TestMtmdBatch:
    @staticmethod
    def _media_chunks(mctx, bitmaps):
        chunks = mctx.tokenize(mctx.marker * len(bitmaps), bitmaps, True, True)
        media = [chunks[i] for i in range(len(chunks)) if chunks[i].type != cy.MtmdInputChunkType.TEXT]
        return chunks, media

    @staticmethod
    def _image(seed):
        side = 64
        return cy.MtmdBitmap.create_image(side, side, bytes((seed + i) % 256 for i in range(side * side * 3)))

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

    def test_chunk_view_keeps_container_alive(self, vision):
        _, mctx = vision
        chunks = mctx.tokenize(mctx.marker, [self._image(0)], True, True)
        before = sys.getrefcount(chunks)
        view = chunks[len(chunks) - 1]
        assert sys.getrefcount(chunks) == before + 1
        n_tokens = view.n_tokens
        del chunks
        gc.collect()
        assert view.n_tokens == n_tokens

    def test_batch_keeps_chunks_alive(self, vision):
        model, mctx = vision

        def first_media(chunks):
            return next(chunks[i] for i in range(len(chunks)) if chunks[i].type != cy.MtmdInputChunkType.TEXT)

        kept = mctx.tokenize(mctx.marker, [self._image(7)], True, True)
        with mctx.batch() as batch:
            batch.add_chunk(first_media(kept))
            batch.encode()
            expected = batch.get_output_embd(first_media(kept), model.n_embd)

        batch = mctx.batch()
        chunks = mctx.tokenize(mctx.marker, [self._image(7)], True, True)
        chunk = first_media(chunks)
        before = sys.getrefcount(chunk)
        batch.add_chunk(chunk)
        assert sys.getrefcount(chunk) == before + 1
        # drop the container; before the fix its chunks were freed here
        del chunks
        gc.collect()
        churn = [mctx.tokenize(mctx.marker, [self._image(i)], True, True) for i in range(8)]
        batch.encode()
        assert batch.get_output_embd(chunk, model.n_embd) == expected
        batch.close()
        assert sys.getrefcount(chunk) == before
        del churn, chunk, kept
        gc.collect()

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
