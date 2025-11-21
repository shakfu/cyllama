"""
Tests for multimodal (mtmd) functionality.

These tests cover the mtmd integration for vision and audio processing.
Note: Some tests require actual model files and may be skipped if not available.
"""

import pytest
import os
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import io

#pytest.skip("Skipping test_mtmd test for now", allow_module_level=True)


try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

import cyllama
from cyllama.llama.mtmd import (
    MtmdContext,
    MtmdContextParams,
    MtmdBitmap,
    MtmdInputChunk,
    MtmdInputChunks,
    MtmdInputChunkType,
    get_default_media_marker,
    MultimodalProcessor,
    VisionLanguageChat,
    AudioProcessor,
    ImageAnalyzer,
    MultimodalError,
    UnsupportedModalityError,
)


class TestMtmdContextParams:
    """Test MtmdContextParams class."""

    def test_default_params(self):
        """Test default parameter initialization."""
        params = MtmdContextParams()
        assert params.use_gpu is True
        assert params.print_timings is False
        assert params.n_threads == 1

    def test_custom_params(self):
        """Test custom parameter initialization."""
        params = MtmdContextParams(
            use_gpu=False,
            print_timings=True,
            n_threads=4
        )
        assert params.use_gpu is False
        assert params.print_timings is True
        assert params.n_threads == 4

    def test_parameter_modification(self):
        """Test parameter modification after creation."""
        params = MtmdContextParams()
        params.use_gpu = False
        params.n_threads = 8
        assert params.use_gpu is False
        assert params.n_threads == 8


class TestMtmdBitmap:
    """Test MtmdBitmap class."""

    def test_create_image_bitmap(self):
        """Test creating image bitmap from RGB data."""
        width, height = 4, 4
        # Create simple RGB data (red square)
        rgb_data = b'\xff\x00\x00' * (width * height)

        bitmap = MtmdBitmap.create_image(width, height, rgb_data)

        assert bitmap.width == width
        assert bitmap.height == height
        assert not bitmap.is_audio
        assert len(bitmap.data) == len(rgb_data)

    def test_create_audio_bitmap(self):
        """Test creating audio bitmap from float samples."""
        samples = [0.1, 0.2, -0.1, -0.2, 0.0] * 100  # 500 samples

        bitmap = MtmdBitmap.create_audio(samples)

        assert bitmap.is_audio
        # Audio bitmap dimensions might be used differently
        assert len(bitmap.data) > 0

    @pytest.mark.skipif(not HAS_PIL, reason="PIL not available")
    def test_create_bitmap_from_pil_image(self):
        """Test creating bitmap from PIL Image."""
        # Create a simple test image
        width, height = 10, 10
        image = Image.new('RGB', (width, height), color='red')

        # Convert to RGB data
        rgb_data = image.tobytes()

        bitmap = MtmdBitmap.create_image(width, height, rgb_data)
        assert bitmap.width == width
        assert bitmap.height == height

    def test_bitmap_id(self):
        """Test bitmap ID functionality."""
        bitmap = MtmdBitmap.create_image(2, 2, b'\x00' * 12)

        # Initially empty ID
        assert bitmap.id == ""

        # Set ID
        test_id = "test_image_123"
        bitmap.id = test_id
        assert bitmap.id == test_id

    def test_invalid_image_data(self):
        """Test error handling for invalid image data."""
        with pytest.raises(OverflowError):
            # Negative dimensions should raise OverflowError
            MtmdBitmap.create_image(-1, 5, b'\x00' * 15)

    def test_empty_audio_samples(self):
        """Test creating bitmap with empty audio samples."""
        bitmap = MtmdBitmap.create_audio([])
        assert bitmap.is_audio


class TestMtmdInputChunks:
    """Test MtmdInputChunks class."""

    def test_empty_chunks(self):
        """Test empty input chunks."""
        chunks = MtmdInputChunks()
        assert len(chunks) == 0
        assert chunks.total_tokens == 0
        assert chunks.total_positions == 0

    def test_chunks_indexing(self):
        """Test chunks indexing behavior."""
        chunks = MtmdInputChunks()

        # Empty chunks should raise IndexError
        with pytest.raises(IndexError):
            _ = chunks[0]

        with pytest.raises(IndexError):
            _ = chunks[-1]


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_default_media_marker(self):
        """Test getting default media marker."""
        marker = get_default_media_marker()
        assert isinstance(marker, str)
        assert len(marker) > 0
        # Should be the expected default marker
        assert marker == "<__media__>"


class TestMultimodalProcessor:
    """Test high-level MultimodalProcessor class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock LlamaModel."""
        model = Mock()
        model.model_ptr = 0x12345678  # Fake pointer
        return model

    @pytest.fixture
    def mock_mtmd_context(self):
        """Create a mock MtmdContext."""
        with patch('cyllama.llama.mtmd.multimodal.MtmdContext') as mock:
            ctx = Mock()
            ctx.supports_vision = True
            ctx.supports_audio = False
            ctx.audio_bitrate = -1
            mock.return_value = ctx
            yield ctx

    def test_processor_initialization(self, mock_model, mock_mtmd_context):
        """Test processor initialization."""
        with patch('os.path.exists', return_value=True):
            processor = MultimodalProcessor("test.mmproj", mock_model)

            assert processor.supports_vision is True
            assert processor.supports_audio is False
            assert processor.audio_bitrate == -1

    def test_process_image_unsupported(self, mock_model):
        """Test processing image when vision is not supported."""
        with patch('cyllama.llama.mtmd.multimodal.MtmdContext') as mock_ctx:
            ctx = Mock()
            ctx.supports_vision = False
            mock_ctx.return_value = ctx

            with patch('os.path.exists', return_value=True):
                processor = MultimodalProcessor("test.mmproj", mock_model)

                with pytest.raises(UnsupportedModalityError):
                    processor.process_image("What is this?", "test.jpg")

    def test_process_audio_unsupported(self, mock_model):
        """Test processing audio when audio is not supported."""
        with patch('cyllama.llama.mtmd.multimodal.MtmdContext') as mock_ctx:
            ctx = Mock()
            ctx.supports_vision = True
            ctx.supports_audio = False
            mock_ctx.return_value = ctx

            with patch('os.path.exists', return_value=True):
                processor = MultimodalProcessor("test.mmproj", mock_model)

                with pytest.raises(UnsupportedModalityError):
                    processor.process_audio("What is this?", "test.wav")

    def test_text_marker_insertion(self, mock_model, mock_mtmd_context):
        """Test automatic marker insertion in text."""
        with patch('os.path.exists', return_value=True):
            processor = MultimodalProcessor("test.mmproj", mock_model)

            # Mock the bitmap creation and tokenization
            mock_bitmap = Mock()
            with patch.object(processor, '_load_image_bitmap', return_value=mock_bitmap):
                with patch.object(processor.mtmd_ctx, 'tokenize') as mock_tokenize:
                    mock_chunks = Mock()
                    mock_tokenize.return_value = mock_chunks

                    # Text without marker should have marker added
                    result = processor.process_image("What is this?", "test.jpg")

                    # Check that tokenize was called with text containing marker
                    args, kwargs = mock_tokenize.call_args
                    text_arg = args[0]
                    assert get_default_media_marker() in text_arg


class TestVisionLanguageChat:
    """Test VisionLanguageChat class."""

    @pytest.fixture
    def mock_setup(self):
        """Set up mocks for VisionLanguageChat tests."""
        model = Mock()
        context = Mock()

        with patch('cyllama.llama.mtmd.multimodal.MultimodalProcessor') as mock_proc:
            proc_instance = Mock()
            proc_instance.supports_vision = True
            mock_proc.return_value = proc_instance

            with patch('os.path.exists', return_value=True):
                chat = VisionLanguageChat("test.mmproj", model, context)
                yield chat, proc_instance

    def test_chat_initialization_vision_unsupported(self):
        """Test chat initialization when vision is not supported."""
        model = Mock()
        context = Mock()

        with patch('cyllama.llama.mtmd.multimodal.MultimodalProcessor') as mock_proc:
            proc_instance = Mock()
            proc_instance.supports_vision = False
            mock_proc.return_value = proc_instance

            with patch('os.path.exists', return_value=True):
                with pytest.raises(UnsupportedModalityError):
                    VisionLanguageChat("test.mmproj", model, context)

    def test_ask_about_image(self, mock_setup):
        """Test asking about an image."""
        chat, processor = mock_setup

        # Mock the image processing
        mock_chunks = Mock()
        processor.process_image.return_value = mock_chunks
        processor.mtmd_ctx.eval_chunks.return_value = 10

        question = "What's in this image?"
        response = chat.ask_about_image(question, "test.jpg")

        assert isinstance(response, str)
        assert len(chat.conversation_history) == 1
        assert chat.conversation_history[0]['question'] == question

    def test_conversation_history_tracking(self, mock_setup):
        """Test conversation history tracking."""
        chat, processor = mock_setup

        # Mock responses
        mock_chunks = Mock()
        processor.process_image.return_value = mock_chunks
        processor.mtmd_ctx.eval_chunks.return_value = 10

        # Ask multiple questions
        chat.ask_about_image("First question?", "image1.jpg")
        chat.continue_conversation("Follow-up question")

        assert len(chat.conversation_history) == 2

        # Clear history
        chat.clear_history()
        assert len(chat.conversation_history) == 0


class TestAudioProcessor:
    """Test AudioProcessor class."""

    def test_initialization_audio_unsupported(self):
        """Test initialization when audio is not supported."""
        model = Mock()

        with patch('cyllama.llama.mtmd.multimodal.MultimodalProcessor') as mock_proc:
            proc_instance = Mock()
            proc_instance.supports_audio = False
            mock_proc.return_value = proc_instance

            with patch('os.path.exists', return_value=True):
                with pytest.raises(UnsupportedModalityError):
                    AudioProcessor("test.mmproj", model)

    def test_audio_processor_supported(self):
        """Test audio processor when audio is supported."""
        model = Mock()

        with patch('cyllama.llama.mtmd.multimodal.MultimodalProcessor') as mock_proc:
            proc_instance = Mock()
            proc_instance.supports_audio = True
            mock_proc.return_value = proc_instance

            with patch('os.path.exists', return_value=True):
                processor = AudioProcessor("test.mmproj", model)
                assert processor.processor.supports_audio


class TestImageAnalyzer:
    """Test ImageAnalyzer class."""

    def test_initialization_vision_unsupported(self):
        """Test initialization when vision is not supported."""
        model = Mock()

        with patch('cyllama.llama.mtmd.multimodal.MultimodalProcessor') as mock_proc:
            proc_instance = Mock()
            proc_instance.supports_vision = False
            mock_proc.return_value = proc_instance

            with patch('os.path.exists', return_value=True):
                with pytest.raises(UnsupportedModalityError):
                    ImageAnalyzer("test.mmproj", model)

    def test_image_analyzer_supported(self):
        """Test image analyzer when vision is supported."""
        model = Mock()

        with patch('cyllama.llama.mtmd.multimodal.MultimodalProcessor') as mock_proc:
            proc_instance = Mock()
            proc_instance.supports_vision = True
            mock_proc.return_value = proc_instance

            with patch('os.path.exists', return_value=True):
                analyzer = ImageAnalyzer("test.mmproj", model)
                assert analyzer.processor.supports_vision

    def test_detail_level_prompts(self):
        """Test different detail level prompts."""
        model = Mock()

        with patch('cyllama.llama.mtmd.multimodal.MultimodalProcessor') as mock_proc:
            proc_instance = Mock()
            proc_instance.supports_vision = True
            proc_instance.process_image.return_value = Mock()
            mock_proc.return_value = proc_instance

            with patch('os.path.exists', return_value=True):
                analyzer = ImageAnalyzer("test.mmproj", model)

                # Test different detail levels
                analyzer.describe_image("test.jpg", "brief")
                analyzer.describe_image("test.jpg", "medium")
                analyzer.describe_image("test.jpg", "detailed")

                # Should have called process_image 3 times
                assert proc_instance.process_image.call_count == 3


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_file_not_found_error(self):
        """Test handling of missing files."""
        model = Mock()

        with patch('os.path.exists', return_value=False):
            with pytest.raises(FileNotFoundError):
                MultimodalProcessor("nonexistent.mmproj", model)

    def test_bitmap_file_loading_error(self):
        """Test bitmap loading error handling."""
        # This would need actual mtmd context to test properly
        # For now, just verify the error paths exist
        with pytest.raises(FileNotFoundError):
            MtmdBitmap.from_file(None, "nonexistent_file.jpg")

    def test_multimodal_error_inheritance(self):
        """Test custom exception inheritance."""
        assert issubclass(UnsupportedModalityError, MultimodalError)
        assert issubclass(MultimodalError, Exception)

    def test_invalid_media_type_handling(self):
        """Test handling of invalid media types."""
        model = Mock()

        # Mock the MtmdContext constructor to avoid type checking issues
        with patch('cyllama.llama.mtmd.multimodal.MtmdContext') as mock_mtmd_ctx:
            mock_ctx_instance = Mock()
            mock_ctx_instance.supports_vision = True
            mock_ctx_instance.supports_audio = False
            mock_ctx_instance.audio_bitrate = -1
            mock_mtmd_ctx.return_value = mock_ctx_instance

            with patch('os.path.exists', return_value=True):
                processor = MultimodalProcessor("test.mmproj", model)

                # Test invalid image type
                with pytest.raises(ValueError):
                    processor._load_image_bitmap(123)  # Invalid type

                # Test invalid audio type
                with pytest.raises(ValueError):
                    processor._load_audio_bitmap({"invalid": "type"})


@pytest.mark.integration
class TestMtmdIntegration:
    """Integration tests that require actual models (marked as integration)."""

    @pytest.fixture
    def model_files(self):
        """Check for required model files."""
        model_dir = Path("models")
        if not model_dir.exists():
            pytest.skip("Models directory not found")

        # Look for any .gguf model file
        model_files = list(model_dir.glob("*.gguf"))
        if not model_files:
            pytest.skip("No GGUF model files found")

        # Look for any .mmproj file
        mmproj_files = list(model_dir.glob("*.mmproj"))
        if not mmproj_files:
            pytest.skip("No multimodal projector files found")

        return model_files[0], mmproj_files[0]

    @pytest.mark.skip(reason="Requires actual model files and may be slow")
    def test_real_model_loading(self, model_files):
        """Test loading real model files."""
        model_path, mmproj_path = model_files

        # Load actual model
        model = cyllama.LlamaModel(str(model_path))

        # Create multimodal processor
        processor = MultimodalProcessor(str(mmproj_path), model)

        # Basic capability checks
        assert isinstance(processor.supports_vision, bool)
        assert isinstance(processor.supports_audio, bool)

    @pytest.mark.skip(reason="Requires actual model files and test images")
    def test_real_image_processing(self, model_files):
        """Test processing real images."""
        model_path, mmproj_path = model_files

        # This would require actual test images and working models
        # Implementation would depend on available test assets
        pass


if __name__ == "__main__":
    pytest.main([__file__])