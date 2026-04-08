"""
Tests for cyllama.__main__ CLI module.

Tests the unified CLI dispatcher including argument parsing, command routing,
prompt input handling, and delegation to sub-module CLIs.
"""

import argparse
import json
import os
import tempfile

import pytest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse(argv):
    """Run main()'s argparse on a given argv list, return parsed args."""
    from cyllama.__main__ import main

    with patch("sys.argv", ["cyllama"] + argv):
        # We can't call main() directly for parse-only tests because it
        # dispatches immediately.  Instead, replicate the parser setup.
        parser = argparse.ArgumentParser(prog="cyllama")
        subparsers = parser.add_subparsers(dest="command")
        subparsers.add_parser("info")
        subparsers.add_parser("version")

        gen_parser = subparsers.add_parser("generate", aliases=["gen"])
        gen_parser.add_argument("-m", "--model", required=True)
        gen_parser.add_argument("-p", "--prompt")
        gen_parser.add_argument("-f", "--file")
        gen_parser.add_argument("-n", "--max-tokens", type=int, default=512)
        gen_parser.add_argument("--temperature", type=float, default=0.8)
        gen_parser.add_argument("--top-k", type=int, default=40)
        gen_parser.add_argument("--top-p", type=float, default=0.95)
        gen_parser.add_argument("--min-p", type=float, default=0.05)
        gen_parser.add_argument("--repeat-penalty", type=float, default=1.1)
        gen_parser.add_argument("-ngl", "--n-gpu-layers", type=int, default=99)
        gen_parser.add_argument("-c", "--ctx-size", type=int, default=None)
        gen_parser.add_argument("--seed", type=int, default=-1)
        gen_parser.add_argument("--stream", action="store_true")
        gen_parser.add_argument("--json", action="store_true")
        gen_parser.add_argument("--verbose", action="store_true")

        chat_parser = subparsers.add_parser("chat")
        chat_parser.add_argument("-m", "--model", required=True)
        chat_parser.add_argument("-p", "--prompt")
        chat_parser.add_argument("-s", "--system")
        chat_parser.add_argument("--template")
        chat_parser.add_argument("-n", "--max-tokens", type=int, default=512)
        chat_parser.add_argument("--temperature", type=float, default=0.8)
        chat_parser.add_argument("--top-k", type=int, default=40)
        chat_parser.add_argument("--top-p", type=float, default=0.95)
        chat_parser.add_argument("--min-p", type=float, default=0.05)
        chat_parser.add_argument("--repeat-penalty", type=float, default=1.1)
        chat_parser.add_argument("-ngl", "--n-gpu-layers", type=int, default=99)
        chat_parser.add_argument("-c", "--ctx-size", type=int, default=2048)
        chat_parser.add_argument("--seed", type=int, default=-1)
        chat_parser.add_argument("--stream", action="store_true")
        chat_parser.add_argument("--json", action="store_true")
        chat_parser.add_argument("--verbose", action="store_true")

        embed_parser = subparsers.add_parser("embed")
        embed_parser.add_argument("-m", "--model", required=True)
        embed_parser.add_argument("-t", "--text", action="append")
        embed_parser.add_argument("-f", "--file")
        embed_parser.add_argument("-ngl", "--n-gpu-layers", type=int, default=99)
        embed_parser.add_argument("-c", "--ctx-size", type=int, default=512)
        embed_parser.add_argument("--pooling", default="mean", choices=["mean", "cls", "last"])
        embed_parser.add_argument("--no-normalize", action="store_true")

        args, _ = parser.parse_known_args(argv)
        return args


# ---------------------------------------------------------------------------
# Argument Parsing Tests
# ---------------------------------------------------------------------------


class TestArgParsing:
    """Test argparse subcommand and option parsing."""

    def test_no_command_shows_help(self, capsys):
        from cyllama.__main__ import main

        with patch("sys.argv", ["cyllama"]):
            ret = main()
        assert ret == 0
        assert "cyllama" in capsys.readouterr().out

    def test_info_command(self):
        args = _parse(["info"])
        assert args.command == "info"

    def test_version_command(self):
        args = _parse(["version"])
        assert args.command == "version"

    def test_generate_defaults(self):
        args = _parse(["generate", "-m", "model.gguf", "-p", "hello"])
        assert args.command == "generate"
        assert args.model == "model.gguf"
        assert args.prompt == "hello"
        assert args.max_tokens == 512
        assert args.temperature == 0.8
        assert args.top_k == 40
        assert args.top_p == 0.95
        assert args.min_p == 0.05
        assert args.repeat_penalty == 1.1
        assert args.n_gpu_layers == 99
        assert args.ctx_size is None
        assert args.seed == -1
        assert args.stream is False
        assert args.json is False
        assert args.verbose is False

    def test_gen_alias(self):
        args = _parse(["gen", "-m", "model.gguf", "-p", "hello"])
        assert args.command == "gen"

    def test_generate_custom_params(self):
        args = _parse(
            [
                "generate",
                "-m",
                "model.gguf",
                "-p",
                "hi",
                "-n",
                "256",
                "--temperature",
                "0.5",
                "--top-k",
                "20",
                "--top-p",
                "0.9",
                "--min-p",
                "0.1",
                "--repeat-penalty",
                "1.2",
                "-ngl",
                "32",
                "-c",
                "4096",
                "--seed",
                "42",
                "--stream",
                "--json",
                "--verbose",
            ]
        )
        assert args.max_tokens == 256
        assert args.temperature == 0.5
        assert args.top_k == 20
        assert args.top_p == 0.9
        assert args.min_p == 0.1
        assert args.repeat_penalty == 1.2
        assert args.n_gpu_layers == 32
        assert args.ctx_size == 4096
        assert args.seed == 42
        assert args.stream is True
        assert args.json is True
        assert args.verbose is True

    def test_generate_file_prompt(self):
        args = _parse(["generate", "-m", "model.gguf", "-f", "prompt.txt"])
        assert args.file == "prompt.txt"
        assert args.prompt is None

    def test_chat_defaults(self):
        args = _parse(["chat", "-m", "model.gguf"])
        assert args.command == "chat"
        assert args.model == "model.gguf"
        assert args.prompt is None
        assert args.system is None
        assert args.template is None
        assert args.max_tokens == 512
        assert args.ctx_size == 2048

    def test_chat_single_turn(self):
        args = _parse(["chat", "-m", "m.gguf", "-p", "hi", "-s", "You are helpful"])
        assert args.prompt == "hi"
        assert args.system == "You are helpful"

    def test_chat_template(self):
        args = _parse(["chat", "-m", "m.gguf", "--template", "chatml"])
        assert args.template == "chatml"

    def test_embed_defaults(self):
        args = _parse(["embed", "-m", "emb.gguf", "-t", "hello"])
        assert args.command == "embed"
        assert args.text == ["hello"]
        assert args.n_gpu_layers == 99

    def test_embed_multiple_texts(self):
        args = _parse(["embed", "-m", "emb.gguf", "-t", "one", "-t", "two"])
        assert args.text == ["one", "two"]


# ---------------------------------------------------------------------------
# cmd_info / cmd_version
# ---------------------------------------------------------------------------


class TestInfoVersion:
    def test_cmd_version(self, capsys):
        from cyllama.__main__ import cmd_version

        cmd_version()
        out = capsys.readouterr().out.strip()
        # Should print a version string
        assert len(out) > 0

    def test_cmd_info(self, capsys):
        from cyllama.__main__ import cmd_info

        cmd_info()
        out = capsys.readouterr().out
        assert "cyllama" in out


# ---------------------------------------------------------------------------
# cmd_generate
# ---------------------------------------------------------------------------


class TestCmdGenerate:
    def _make_args(self, **overrides):
        defaults = dict(
            model="model.gguf",
            prompt="hello",
            file=None,
            max_tokens=32,
            temperature=0.8,
            top_k=40,
            top_p=0.95,
            min_p=0.05,
            repeat_penalty=1.1,
            n_gpu_layers=99,
            ctx_size=None,
            seed=-1,
            stream=False,
            json=False,
            verbose=False,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_no_prompt_no_file_tty(self):
        """Should return 1 when no prompt source is available."""
        from cyllama.__main__ import cmd_generate

        args = self._make_args(prompt=None, file=None)
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            ret = cmd_generate(args)
        assert ret == 1

    def test_prompt_from_file(self, tmp_path):
        """Should read prompt from file."""
        from cyllama.__main__ import cmd_generate

        p = tmp_path / "prompt.txt"
        p.write_text("from file")
        args = self._make_args(prompt=None, file=str(p))

        mock_response = MagicMock()
        mock_response.__str__ = lambda self: "response text"
        with patch("cyllama.api.complete", return_value=mock_response) as mock_complete:
            ret = cmd_generate(args)

        assert ret == 0
        called_prompt = mock_complete.call_args[0][0]
        assert called_prompt == "from file"

    def test_prompt_from_arg(self):
        """Should use -p prompt."""
        from cyllama.__main__ import cmd_generate

        args = self._make_args(prompt="test prompt")

        mock_response = MagicMock()
        with patch("cyllama.api.complete", return_value=mock_response):
            ret = cmd_generate(args)
        assert ret == 0

    def test_stream_mode(self):
        """Should iterate chunks in stream mode."""
        from cyllama.__main__ import cmd_generate

        args = self._make_args(stream=True)

        with patch("cyllama.api.complete", return_value=iter(["hello", " world"])):
            ret = cmd_generate(args)
        assert ret == 0

    def test_json_output(self):
        """Should call to_json() when --json is set."""
        from cyllama.__main__ import cmd_generate

        args = self._make_args(json=True)

        mock_response = MagicMock()
        mock_response.to_json.return_value = '{"text": "hi"}'
        with patch("cyllama.api.complete", return_value=mock_response):
            ret = cmd_generate(args)
        assert ret == 0
        mock_response.to_json.assert_called_once()

    def test_keyboard_interrupt(self):
        """Should catch KeyboardInterrupt and return 130."""
        from cyllama.__main__ import cmd_generate

        args = self._make_args()

        with patch("cyllama.api.complete", side_effect=KeyboardInterrupt):
            ret = cmd_generate(args)
        assert ret == 130

    def test_config_passthrough(self):
        """Should pass sampling parameters to GenerationConfig."""
        from cyllama.__main__ import cmd_generate

        args = self._make_args(
            max_tokens=100,
            temperature=0.5,
            top_k=20,
            top_p=0.9,
            min_p=0.1,
            repeat_penalty=1.3,
            n_gpu_layers=32,
            ctx_size=4096,
            seed=42,
        )

        mock_response = MagicMock()
        with patch("cyllama.api.complete", return_value=mock_response) as mock_complete:
            cmd_generate(args)

        config = mock_complete.call_args[0][2]  # third positional arg
        assert config.max_tokens == 100
        assert config.temperature == 0.5
        assert config.top_k == 20
        assert config.top_p == 0.9
        assert config.min_p == 0.1
        assert config.repeat_penalty == 1.3
        assert config.n_gpu_layers == 32
        assert config.n_ctx == 4096
        assert config.seed == 42


# ---------------------------------------------------------------------------
# cmd_chat
# ---------------------------------------------------------------------------


class TestCmdChat:
    def _make_args(self, **overrides):
        defaults = dict(
            model="model.gguf",
            prompt=None,
            system=None,
            template=None,
            max_tokens=512,
            temperature=0.8,
            top_k=40,
            top_p=0.95,
            min_p=0.05,
            repeat_penalty=1.1,
            n_gpu_layers=99,
            ctx_size=2048,
            seed=-1,
            stream=False,
            json=False,
            verbose=False,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_single_turn(self):
        """Single-turn chat with -p."""
        from cyllama.__main__ import cmd_chat

        args = self._make_args(prompt="What is Python?")

        mock_response = MagicMock()
        with patch("cyllama.api.chat", return_value=mock_response) as mock_chat:
            ret = cmd_chat(args)

        assert ret == 0
        messages = mock_chat.call_args[0][0]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What is Python?"

    def test_single_turn_with_system(self):
        """Single-turn chat with system prompt."""
        from cyllama.__main__ import cmd_chat

        args = self._make_args(prompt="hi", system="You are helpful")

        mock_response = MagicMock()
        with patch("cyllama.api.chat", return_value=mock_response) as mock_chat:
            ret = cmd_chat(args)

        messages = mock_chat.call_args[0][0]
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "You are helpful"}
        assert messages[1] == {"role": "user", "content": "hi"}

    def test_single_turn_stream(self):
        """Stream mode for single-turn chat."""
        from cyllama.__main__ import cmd_chat

        args = self._make_args(prompt="hi", stream=True)

        with patch("cyllama.api.chat", return_value=iter(["hello"])):
            ret = cmd_chat(args)
        assert ret == 0

    def test_single_turn_json(self):
        """JSON output for single-turn chat."""
        from cyllama.__main__ import cmd_chat

        args = self._make_args(prompt="hi", json=True)

        mock_response = MagicMock()
        mock_response.to_json.return_value = '{"text": "hi"}'
        with patch("cyllama.api.chat", return_value=mock_response):
            ret = cmd_chat(args)
        assert ret == 0
        mock_response.to_json.assert_called_once()

    def test_single_turn_template(self):
        """Template passed through to chat()."""
        from cyllama.__main__ import cmd_chat

        args = self._make_args(prompt="hi", template="chatml")

        mock_response = MagicMock()
        with patch("cyllama.api.chat", return_value=mock_response) as mock_chat:
            cmd_chat(args)

        assert mock_chat.call_args[1]["template"] == "chatml"

    def test_interactive_delegates(self):
        """Without -p, should delegate to llama.chat.main()."""
        from cyllama.__main__ import cmd_chat

        args = self._make_args()

        with patch("cyllama.__main__.sys") as mock_sys, patch("cyllama.llama.chat.main") as mock_main:
            mock_sys.argv = ["cyllama", "chat"]
            ret = cmd_chat(args)

        mock_main.assert_called_once()


# ---------------------------------------------------------------------------
# cmd_embed
# ---------------------------------------------------------------------------


class TestCmdEmbed:
    def _make_args(self, **overrides):
        defaults = dict(
            model="emb.gguf",
            text=None,
            file=None,
            n_gpu_layers=99,
            ctx_size=512,
            pooling="mean",
            no_normalize=False,
            dim=False,
            similarity=None,
            threshold=0.0,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_no_text_tty(self):
        """Should return 1 when no text source is available."""
        from cyllama.__main__ import cmd_embed

        args = self._make_args()
        mock_embedder = MagicMock()
        with patch("cyllama.rag.embedder.Embedder", return_value=mock_embedder), patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            ret = cmd_embed(args)
        assert ret == 1

    def _mock_stdin_tty(self):
        """Return a mock stdin that reports isatty()=True and yields nothing."""
        mock = MagicMock()
        mock.isatty.return_value = True
        mock.__iter__ = MagicMock(return_value=iter([]))
        return mock

    def test_text_from_arg(self, capsys):
        """Should embed texts from -t."""
        from cyllama.__main__ import cmd_embed
        import numpy as np

        args = self._make_args(text=["hello", "world"])

        mock_embedder = MagicMock()
        mock_embedder.embed_batch.return_value = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
        with (
            patch("cyllama.rag.embedder.Embedder", return_value=mock_embedder),
            patch("sys.stdin", self._mock_stdin_tty()),
        ):
            ret = cmd_embed(args)

        assert ret == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert len(data) == 2

    def test_text_from_file(self, tmp_path, capsys):
        """Should read texts from file, one per line."""
        from cyllama.__main__ import cmd_embed
        import numpy as np

        f = tmp_path / "texts.txt"
        f.write_text("line one\nline two\n\n")  # blank line should be skipped
        args = self._make_args(file=str(f))

        mock_embedder = MagicMock()
        mock_embedder.embed_batch.return_value = [np.array([0.1]), np.array([0.2])]
        with (
            patch("cyllama.rag.embedder.Embedder", return_value=mock_embedder),
            patch("sys.stdin", self._mock_stdin_tty()),
        ):
            ret = cmd_embed(args)

        assert ret == 0
        texts_passed = mock_embedder.embed_batch.call_args[0][0]
        assert texts_passed == ["line one", "line two"]

    def test_dim(self, capsys):
        """--dim should print embedding dimensions and exit."""
        from cyllama.__main__ import cmd_embed

        args = self._make_args(dim=True)

        mock_embedder = MagicMock()
        mock_embedder.dimension = 384
        with patch("cyllama.rag.embedder.Embedder", return_value=mock_embedder):
            ret = cmd_embed(args)

        assert ret == 0
        assert capsys.readouterr().out.strip() == "384"

    def test_similarity(self, capsys):
        """--similarity should rank texts by cosine similarity."""
        from cyllama.__main__ import cmd_embed

        args = self._make_args(text=["cats are great", "dogs are fun", "quantum physics"], similarity="I love cats")

        mock_embedder = MagicMock()
        # query embedding
        mock_embedder.embed.return_value = [1.0, 0.0, 0.0]
        # text embeddings: first is most similar to query
        mock_embedder.embed_batch.return_value = [
            [0.9, 0.1, 0.0],  # cats - high similarity
            [0.5, 0.5, 0.0],  # dogs - medium
            [0.0, 0.0, 1.0],  # physics - low
        ]
        with (
            patch("cyllama.rag.embedder.Embedder", return_value=mock_embedder),
            patch("sys.stdin", self._mock_stdin_tty()),
        ):
            ret = cmd_embed(args)

        assert ret == 0
        lines = capsys.readouterr().out.strip().split("\n")
        assert len(lines) == 3
        # First result should be the most similar (cats)
        assert "cats are great" in lines[0]
        # Last result should be the least similar (physics)
        assert "quantum physics" in lines[2]

    def test_similarity_no_texts(self):
        """--similarity without texts should error."""
        from cyllama.__main__ import cmd_embed

        args = self._make_args(similarity="query")

        mock_embedder = MagicMock()
        with patch("cyllama.rag.embedder.Embedder", return_value=mock_embedder), patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            ret = cmd_embed(args)
        assert ret == 1

    def test_similarity_threshold(self, capsys):
        """--threshold should filter out low-scoring results."""
        from cyllama.__main__ import cmd_embed

        args = self._make_args(text=["cats", "dogs", "physics"], similarity="cats", threshold=0.8)

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [1.0, 0.0, 0.0]
        mock_embedder.embed_batch.return_value = [
            [0.95, 0.05, 0.0],  # cats - above threshold
            [0.5, 0.5, 0.0],  # dogs - below threshold
            [0.0, 0.0, 1.0],  # physics - below threshold
        ]
        with (
            patch("cyllama.rag.embedder.Embedder", return_value=mock_embedder),
            patch("sys.stdin", self._mock_stdin_tty()),
        ):
            ret = cmd_embed(args)

        assert ret == 0
        lines = capsys.readouterr().out.strip().split("\n")
        assert len(lines) == 1
        assert "cats" in lines[0]

    def test_pooling_passthrough(self):
        """--pooling should be passed to Embedder."""
        from cyllama.__main__ import cmd_embed

        args = self._make_args(text=["hello"], pooling="cls", dim=True)

        mock_embedder = MagicMock()
        mock_embedder.dimension = 384
        with patch("cyllama.rag.embedder.Embedder", return_value=mock_embedder) as mock_cls:
            cmd_embed(args)

        assert mock_cls.call_args[1]["pooling"] == "cls"

    def test_no_normalize_passthrough(self):
        """--no-normalize should pass normalize=False to Embedder."""
        from cyllama.__main__ import cmd_embed

        args = self._make_args(text=["hello"], no_normalize=True, dim=True)

        mock_embedder = MagicMock()
        mock_embedder.dimension = 384
        with patch("cyllama.rag.embedder.Embedder", return_value=mock_embedder) as mock_cls:
            cmd_embed(args)

        assert mock_cls.call_args[1]["normalize"] is False

    def test_normalize_default(self):
        """Without --no-normalize, normalize should be True."""
        from cyllama.__main__ import cmd_embed

        args = self._make_args(text=["hello"], dim=True)

        mock_embedder = MagicMock()
        mock_embedder.dimension = 384
        with patch("cyllama.rag.embedder.Embedder", return_value=mock_embedder) as mock_cls:
            cmd_embed(args)

        assert mock_cls.call_args[1]["normalize"] is True


# ---------------------------------------------------------------------------
# cmd_rag
# ---------------------------------------------------------------------------


class TestCmdRag:
    def _make_args(self, **overrides):
        defaults = dict(
            model="llm.gguf",
            embedding_model="emb.gguf",
            file=None,
            dir=None,
            glob="**/*",
            prompt=None,
            system=None,
            max_tokens=512,
            temperature=0.7,
            top_k=5,
            threshold=None,
            n_gpu_layers=99,
            stream=False,
            sources=False,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_no_documents(self):
        """Should return 1 when no files or dirs provided."""
        from cyllama.__main__ import cmd_rag

        args = self._make_args()

        mock_rag = MagicMock()
        mock_rag.add_documents.return_value = []
        with patch("cyllama.rag.RAG", return_value=mock_rag):
            ret = cmd_rag(args)
        assert ret == 1

    def test_single_query(self, capsys):
        """Single query with -p."""
        from cyllama.__main__ import cmd_rag

        args = self._make_args(file=["corpus.txt"], prompt="What is this about?")

        mock_rag = MagicMock()
        mock_rag.add_documents.return_value = [1, 2, 3]
        mock_response = MagicMock()
        mock_response.text = "It is about testing."
        mock_response.sources = []
        mock_rag.query.return_value = mock_response

        with patch("cyllama.rag.RAG", return_value=mock_rag):
            ret = cmd_rag(args)

        assert ret == 0
        out = capsys.readouterr().out
        assert "It is about testing." in out

    def test_single_query_stream(self, capsys):
        """Streaming single query."""
        from cyllama.__main__ import cmd_rag

        args = self._make_args(file=["corpus.txt"], prompt="question", stream=True)

        mock_rag = MagicMock()
        mock_rag.add_documents.return_value = [1, 2]
        mock_rag.stream.return_value = iter(["hello", " world"])

        with patch("cyllama.rag.RAG", return_value=mock_rag):
            ret = cmd_rag(args)

        assert ret == 0
        out = capsys.readouterr().out
        assert "hello world" in out

    def test_single_query_with_sources(self, capsys):
        """Single query with --sources."""
        from cyllama.__main__ import cmd_rag

        args = self._make_args(file=["corpus.txt"], prompt="question", sources=True)

        mock_rag = MagicMock()
        mock_rag.add_documents.return_value = [1]
        mock_source = MagicMock()
        mock_source.score = 0.85
        mock_source.text = "relevant chunk text here"
        mock_response = MagicMock()
        mock_response.text = "answer"
        mock_response.sources = [mock_source]
        mock_rag.query.return_value = mock_response

        with patch("cyllama.rag.RAG", return_value=mock_rag):
            ret = cmd_rag(args)

        assert ret == 0
        out = capsys.readouterr().out
        assert "Sources" in out
        assert "0.8500" in out

    def test_keyboard_interrupt(self):
        """Should handle KeyboardInterrupt gracefully."""
        from cyllama.__main__ import cmd_rag

        args = self._make_args(file=["corpus.txt"], prompt="question")

        mock_rag = MagicMock()
        mock_rag.add_documents.return_value = [1]
        mock_rag.query.side_effect = KeyboardInterrupt

        with patch("cyllama.rag.RAG", return_value=mock_rag):
            ret = cmd_rag(args)
        assert ret == 130


# ---------------------------------------------------------------------------
# _delegate
# ---------------------------------------------------------------------------


class TestDelegate:
    def test_delegate_rewrites_argv(self):
        """Should strip subcommand and call module's main()."""
        from cyllama.__main__ import _delegate

        mock_main = MagicMock()

        with (
            patch("sys.argv", ["cyllama", "server", "--port", "8080"]),
            patch("importlib.import_module") as mock_import,
        ):
            mock_mod = MagicMock()
            mock_mod.main = mock_main
            mock_import.return_value = mock_mod
            _delegate(".llama.server.__main__")

        mock_main.assert_called_once()


# ---------------------------------------------------------------------------
# main() dispatch
# ---------------------------------------------------------------------------


class TestMainDispatch:
    def test_info_dispatch(self):
        from cyllama.__main__ import main

        with patch("sys.argv", ["cyllama", "info"]), patch("cyllama.__main__.cmd_info") as mock:
            main()
        mock.assert_called_once()

    def test_version_dispatch(self):
        from cyllama.__main__ import main

        with patch("sys.argv", ["cyllama", "version"]), patch("cyllama.__main__.cmd_version") as mock:
            main()
        mock.assert_called_once()

    def test_generate_dispatch(self):
        from cyllama.__main__ import main

        with (
            patch("sys.argv", ["cyllama", "generate", "-m", "m.gguf", "-p", "hi"]),
            patch("cyllama.__main__.cmd_generate", return_value=0) as mock,
        ):
            main()
        mock.assert_called_once()

    def test_gen_alias_dispatch(self):
        from cyllama.__main__ import main

        with (
            patch("sys.argv", ["cyllama", "gen", "-m", "m.gguf", "-p", "hi"]),
            patch("cyllama.__main__.cmd_generate", return_value=0) as mock,
        ):
            main()
        mock.assert_called_once()

    def test_chat_dispatch(self):
        from cyllama.__main__ import main

        with (
            patch("sys.argv", ["cyllama", "chat", "-m", "m.gguf", "-p", "hi"]),
            patch("cyllama.__main__.cmd_chat", return_value=0) as mock,
        ):
            main()
        mock.assert_called_once()

    def test_embed_dispatch(self):
        from cyllama.__main__ import main

        with (
            patch("sys.argv", ["cyllama", "embed", "-m", "m.gguf", "-t", "hi"]),
            patch("cyllama.__main__.cmd_embed", return_value=0) as mock,
        ):
            main()
        mock.assert_called_once()

    def test_rag_dispatch(self):
        from cyllama.__main__ import main

        with (
            patch("sys.argv", ["cyllama", "rag", "-m", "m.gguf", "-e", "e.gguf", "-f", "f.txt"]),
            patch("cyllama.__main__.cmd_rag", return_value=0) as mock,
        ):
            main()
        mock.assert_called_once()

    def test_server_delegates(self):
        from cyllama.__main__ import main

        with (
            patch("sys.argv", ["cyllama", "server", "--port", "8080"]),
            patch("cyllama.__main__._delegate", return_value=0) as mock,
        ):
            main()
        mock.assert_called_once_with(".llama.server.__main__")

    def test_sd_delegates(self):
        from cyllama.__main__ import main

        with (
            patch("sys.argv", ["cyllama", "sd", "txt2img"]),
            patch("cyllama.__main__._delegate", return_value=0) as mock,
        ):
            main()
        mock.assert_called_once_with(".sd.__main__")

    def test_transcribe_delegates(self):
        from cyllama.__main__ import main

        with (
            patch("sys.argv", ["cyllama", "transcribe", "-m", "w.bin"]),
            patch("cyllama.__main__._delegate", return_value=0) as mock,
        ):
            main()
        mock.assert_called_once_with(".whisper.cli")

    def test_tts_delegates(self):
        from cyllama.__main__ import main

        with patch("sys.argv", ["cyllama", "tts"]), patch("cyllama.__main__._delegate", return_value=0) as mock:
            main()
        mock.assert_called_once_with(".llama.tts")

    def test_agent_delegates(self):
        from cyllama.__main__ import main

        with (
            patch("sys.argv", ["cyllama", "agent", "run"]),
            patch("cyllama.__main__._delegate", return_value=0) as mock,
        ):
            main()
        mock.assert_called_once_with(".agents.cli")

    def test_memory_delegates(self):
        from cyllama.__main__ import main

        with patch("sys.argv", ["cyllama", "memory"]), patch("cyllama.__main__._delegate", return_value=0) as mock:
            main()
        mock.assert_called_once_with(".memory")
