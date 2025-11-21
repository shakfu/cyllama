"""
High-level Python wrappers for multimodal functionality.

This module provides convenient Python classes for common multimodal tasks
like vision-language processing and audio analysis.
"""

import os
import io
from typing import List, Optional, Union, Dict, Any, Tuple
from pathlib import Path
import logging

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from ..llama_cpp import (
    MtmdContext,
    MtmdContextParams,
    MtmdBitmap,
    MtmdInputChunks,
    get_default_media_marker,
)

logger = logging.getLogger(__name__)


class MultimodalError(Exception):
    """Base exception for multimodal processing errors."""
    pass


class UnsupportedModalityError(MultimodalError):
    """Raised when a requested modality is not supported by the model."""
    pass


class MultimodalProcessor:
    """High-level multimodal processor for vision and audio tasks."""

    def __init__(self, mmproj_path: str, llama_model, **kwargs):
        """Initialize multimodal processor.

        Args:
            mmproj_path: Path to multimodal projector file (.mmproj)
            llama_model: LlamaModel instance
            **kwargs: Additional parameters passed to MtmdContextParams
        """
        # Check file exists before doing anything else
        if not os.path.exists(mmproj_path):
            raise FileNotFoundError(f"Multimodal projector file not found: {mmproj_path}")

        self.mmproj_path = mmproj_path
        self.llama_model = llama_model

        # Create context parameters
        params = MtmdContextParams(**kwargs)

        # Initialize mtmd context
        self.mtmd_ctx = MtmdContext(mmproj_path, llama_model, params)

        # Cache capabilities
        self._supports_vision = self.mtmd_ctx.supports_vision
        self._supports_audio = self.mtmd_ctx.supports_audio
        self._audio_bitrate = self.mtmd_ctx.audio_bitrate

        logger.info(f"Initialized multimodal processor:")
        logger.info(f"  Vision support: {self._supports_vision}")
        logger.info(f"  Audio support: {self._supports_audio}")
        if self._supports_audio:
            logger.info(f"  Audio bitrate: {self._audio_bitrate} Hz")

    @property
    def supports_vision(self) -> bool:
        """Check if vision processing is supported."""
        return self._supports_vision

    @property
    def supports_audio(self) -> bool:
        """Check if audio processing is supported."""
        return self._supports_audio

    @property
    def audio_bitrate(self) -> int:
        """Get supported audio bitrate in Hz."""
        return self._audio_bitrate

    def process_image(self, text: str, image: Union[str, bytes, 'Image.Image'],
                     add_special: bool = True, parse_special: bool = True) -> MtmdInputChunks:
        """Process text with an image.

        Args:
            text: Input text with media marker (or will be added automatically)
            image: Image file path, bytes, or PIL Image
            add_special: Whether to add special tokens
            parse_special: Whether to parse special tokens

        Returns:
            MtmdInputChunks ready for evaluation

        Raises:
            UnsupportedModalityError: If vision is not supported
            MultimodalError: If processing fails
        """
        if not self.supports_vision:
            raise UnsupportedModalityError("Vision processing not supported by this model")

        # Ensure text contains media marker
        marker = get_default_media_marker()
        if marker not in text:
            text = f"{text} {marker}"

        # Load image as bitmap
        try:
            bitmap = self._load_image_bitmap(image)
            return self.mtmd_ctx.tokenize(text, [bitmap], add_special, parse_special)
        except Exception as e:
            raise MultimodalError(f"Failed to process image: {e}")

    def process_audio(self, text: str, audio: Union[str, bytes, List[float]],
                     add_special: bool = True, parse_special: bool = True) -> MtmdInputChunks:
        """Process text with audio.

        Args:
            text: Input text with media marker (or will be added automatically)
            audio: Audio file path, bytes, or list of float samples
            add_special: Whether to add special tokens
            parse_special: Whether to parse special tokens

        Returns:
            MtmdInputChunks ready for evaluation

        Raises:
            UnsupportedModalityError: If audio is not supported
            MultimodalError: If processing fails
        """
        if not self.supports_audio:
            raise UnsupportedModalityError("Audio processing not supported by this model")

        # Ensure text contains media marker
        marker = get_default_media_marker()
        if marker not in text:
            text = f"{text} {marker}"

        # Load audio as bitmap
        try:
            bitmap = self._load_audio_bitmap(audio)
            return self.mtmd_ctx.tokenize(text, [bitmap], add_special, parse_special)
        except Exception as e:
            raise MultimodalError(f"Failed to process audio: {e}")

    def process_multimodal(self, text: str, media_files: List[Union[str, bytes]],
                          add_special: bool = True, parse_special: bool = True) -> MtmdInputChunks:
        """Process text with multiple media files.

        Args:
            text: Input text with media markers
            media_files: List of media file paths or bytes
            add_special: Whether to add special tokens
            parse_special: Whether to parse special tokens

        Returns:
            MtmdInputChunks ready for evaluation

        Raises:
            MultimodalError: If processing fails
        """
        bitmaps = []

        for media in media_files:
            try:
                # Try to detect media type and load appropriately
                if isinstance(media, str):
                    # File path - determine type by extension or content
                    ext = Path(media).suffix.lower()
                    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                        if not self.supports_vision:
                            raise UnsupportedModalityError("Vision not supported")
                        bitmap = self._load_image_bitmap(media)
                    elif ext in ['.wav', '.mp3', '.flac']:
                        if not self.supports_audio:
                            raise UnsupportedModalityError("Audio not supported")
                        bitmap = self._load_audio_bitmap(media)
                    else:
                        # Try to auto-detect using buffer loading
                        bitmap = MtmdBitmap.from_file(self.mtmd_ctx, media)
                else:
                    # Bytes - try buffer loading (will auto-detect type)
                    bitmap = MtmdBitmap.from_buffer(self.mtmd_ctx, media)

                bitmaps.append(bitmap)
            except Exception as e:
                raise MultimodalError(f"Failed to load media: {e}")

        try:
            return self.mtmd_ctx.tokenize(text, bitmaps, add_special, parse_special)
        except Exception as e:
            raise MultimodalError(f"Failed to tokenize multimodal input: {e}")

    def _load_image_bitmap(self, image: Union[str, bytes, 'Image.Image']) -> MtmdBitmap:
        """Load image as MtmdBitmap."""
        if isinstance(image, str):
            # File path
            return MtmdBitmap.from_file(self.mtmd_ctx, image)
        elif isinstance(image, bytes):
            # Image bytes
            return MtmdBitmap.from_buffer(self.mtmd_ctx, image)
        elif HAS_PIL and isinstance(image, Image.Image):
            # PIL Image
            if image.mode != 'RGB':
                image = image.convert('RGB')

            width, height = image.size
            rgb_data = image.tobytes()
            return MtmdBitmap.create_image(width, height, rgb_data)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def _load_audio_bitmap(self, audio: Union[str, bytes, List[float]]) -> MtmdBitmap:
        """Load audio as MtmdBitmap."""
        if isinstance(audio, str):
            # File path
            return MtmdBitmap.from_file(self.mtmd_ctx, audio)
        elif isinstance(audio, bytes):
            # Audio bytes
            return MtmdBitmap.from_buffer(self.mtmd_ctx, audio)
        elif isinstance(audio, list):
            # Float samples
            return MtmdBitmap.create_audio(audio)
        else:
            raise ValueError(f"Unsupported audio type: {type(audio)}")


class VisionLanguageChat:
    """Specialized class for vision-language conversations."""

    def __init__(self, mmproj_path: str, llama_model, llama_context, **kwargs):
        """Initialize vision-language chat.

        Args:
            mmproj_path: Path to multimodal projector file
            llama_model: LlamaModel instance
            llama_context: LlamaContext instance
            **kwargs: Additional parameters for processor
        """
        self.processor = MultimodalProcessor(mmproj_path, llama_model, **kwargs)
        self.llama_context = llama_context
        self.conversation_history = []

        if not self.processor.supports_vision:
            raise UnsupportedModalityError("Vision not supported by this model")

    def ask_about_image(self, question: str, image: Union[str, bytes, 'Image.Image'],
                       max_tokens: int = 512) -> str:
        """Ask a question about an image.

        Args:
            question: Question to ask about the image
            image: Image to analyze
            max_tokens: Maximum tokens to generate

        Returns:
            Model's response about the image
        """
        # Process the image with the question
        chunks = self.processor.process_image(question, image)

        # Evaluate chunks
        n_past = self.processor.mtmd_ctx.eval_chunks(
            self.llama_context, chunks, n_past=0
        )

        # Generate response (simplified - would need proper generation loop)
        # This is a placeholder that would need to be implemented with proper sampling
        response = "Image analysis response would be generated here"

        # Store in conversation history
        self.conversation_history.append({
            'question': question,
            'image': image,
            'response': response,
            'n_past': n_past
        })

        return response

    def continue_conversation(self, message: str, max_tokens: int = 512) -> str:
        """Continue the conversation without a new image.

        Args:
            message: Follow-up message
            max_tokens: Maximum tokens to generate

        Returns:
            Model's response
        """
        # This would implement conversation continuation
        # using the existing context and history
        response = "Conversation continuation would be implemented here"

        self.conversation_history.append({
            'message': message,
            'response': response
        })

        return response

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history.clear()


class AudioProcessor:
    """Specialized class for audio processing tasks."""

    def __init__(self, mmproj_path: str, llama_model, **kwargs):
        """Initialize audio processor.

        Args:
            mmproj_path: Path to multimodal projector file
            llama_model: LlamaModel instance
            **kwargs: Additional parameters
        """
        self.processor = MultimodalProcessor(mmproj_path, llama_model, **kwargs)

        if not self.processor.supports_audio:
            raise UnsupportedModalityError("Audio not supported by this model")

    def transcribe_audio(self, audio: Union[str, bytes, List[float]]) -> str:
        """Transcribe audio to text.

        Args:
            audio: Audio input

        Returns:
            Transcribed text
        """
        # Process audio for transcription
        prompt = "Transcribe this audio:"
        chunks = self.processor.process_audio(prompt, audio)

        # This would need proper implementation with generation
        return "Audio transcription would be implemented here"

    def analyze_audio(self, question: str, audio: Union[str, bytes, List[float]]) -> str:
        """Analyze audio content based on a question.

        Args:
            question: Question about the audio
            audio: Audio input

        Returns:
            Analysis response
        """
        chunks = self.processor.process_audio(question, audio)

        # This would need proper implementation with generation
        return "Audio analysis would be implemented here"


class ImageAnalyzer:
    """Specialized class for image analysis tasks."""

    def __init__(self, mmproj_path: str, llama_model, **kwargs):
        """Initialize image analyzer.

        Args:
            mmproj_path: Path to multimodal projector file
            llama_model: LlamaModel instance
            **kwargs: Additional parameters
        """
        self.processor = MultimodalProcessor(mmproj_path, llama_model, **kwargs)

        if not self.processor.supports_vision:
            raise UnsupportedModalityError("Vision not supported by this model")

    def describe_image(self, image: Union[str, bytes, 'Image.Image'],
                      detail_level: str = "medium") -> str:
        """Generate a description of an image.

        Args:
            image: Image to describe
            detail_level: Level of detail ("brief", "medium", "detailed")

        Returns:
            Image description
        """
        if detail_level == "brief":
            prompt = "Briefly describe this image:"
        elif detail_level == "detailed":
            prompt = "Provide a detailed description of this image:"
        else:
            prompt = "Describe this image:"

        chunks = self.processor.process_image(prompt, image)

        # This would need proper implementation with generation
        return "Image description would be implemented here"

    def detect_objects(self, image: Union[str, bytes, 'Image.Image']) -> List[str]:
        """Detect objects in an image.

        Args:
            image: Image to analyze

        Returns:
            List of detected objects
        """
        prompt = "List all objects visible in this image:"
        chunks = self.processor.process_image(prompt, image)

        # This would need proper implementation with generation and parsing
        return ["Object detection would be implemented here"]

    def answer_question(self, question: str, image: Union[str, bytes, 'Image.Image']) -> str:
        """Answer a specific question about an image.

        Args:
            question: Question to ask
            image: Image to analyze

        Returns:
            Answer to the question
        """
        chunks = self.processor.process_image(question, image)

        # This would need proper implementation with generation
        return "Question answering would be implemented here"