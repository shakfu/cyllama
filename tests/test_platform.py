"""Tests for platform-specific runtime setup."""

import os
from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def reset_initialized():
    """Reset the module-level _initialized flag between tests."""
    from cyllama.utils import platform as plat_mod

    plat_mod._initialized = False
    yield
    plat_mod._initialized = False


class TestEnsureNativeDeps:
    """Tests for ensure_native_deps()."""

    def test_noop_on_non_windows(self):
        from cyllama.utils import platform as plat_mod

        with mock.patch.object(plat_mod, "sys") as mock_sys:
            mock_sys.platform = "linux"
            plat_mod.ensure_native_deps()
            # Observable side effect: _initialized flips to True even on
            # non-Windows so subsequent calls short-circuit.
            assert plat_mod._initialized is True

    def test_idempotent(self):
        from cyllama.utils import platform as plat_mod

        with mock.patch.object(plat_mod, "sys") as mock_sys:
            mock_sys.platform = "linux"
            plat_mod.ensure_native_deps()
            assert plat_mod._initialized is True

            # Second call should be a no-op (won't even check sys.platform)
            mock_sys.platform = "win32"
            plat_mod.ensure_native_deps()
            # If it weren't idempotent, it would try win32 path on the second call

    def test_noop_when_build_config_missing(self):
        """When build_config.json is missing, backend_enabled() returns False and CUDA setup is skipped."""
        from cyllama.utils import platform as plat_mod
        from cyllama._internal import build_config

        with (
            mock.patch.object(plat_mod, "sys") as mock_sys,
            mock.patch.object(plat_mod, "_setup_cuda_dll_paths") as mock_cuda,
            mock.patch.object(build_config, "backend_enabled", return_value=False),
        ):
            mock_sys.platform = "win32"
            plat_mod.ensure_native_deps()
            mock_cuda.assert_not_called()

    def test_skips_cuda_when_not_enabled(self):
        from cyllama.utils import platform as plat_mod
        from cyllama._internal import build_config

        with (
            mock.patch.object(plat_mod, "sys") as mock_sys,
            mock.patch.object(plat_mod, "_setup_cuda_dll_paths") as mock_cuda,
            mock.patch.object(build_config, "backend_enabled", return_value=False),
        ):
            mock_sys.platform = "win32"
            plat_mod.ensure_native_deps()
            mock_cuda.assert_not_called()

    def test_calls_cuda_setup_when_enabled(self):
        from cyllama.utils import platform as plat_mod
        from cyllama._internal import build_config

        with (
            mock.patch.object(plat_mod, "sys") as mock_sys,
            mock.patch.object(plat_mod, "_setup_cuda_dll_paths") as mock_cuda,
            mock.patch.object(build_config, "backend_enabled", return_value=True),
        ):
            mock_sys.platform = "win32"
            plat_mod.ensure_native_deps()
            mock_cuda.assert_called_once()


class TestSetupCudaDllPaths:
    """Tests for _setup_cuda_dll_paths().

    _setup_cuda_dll_paths() imports os/shutil/glob locally, so we patch
    on the real modules rather than on the platform module namespace.
    """

    def test_noop_without_add_dll_directory(self):
        from cyllama.utils.platform import _setup_cuda_dll_paths

        # Temporarily remove add_dll_directory if it exists
        had_attr = hasattr(os, "add_dll_directory")
        if had_attr:
            original = os.add_dll_directory
            delattr(os, "add_dll_directory")
        try:
            # The function must tolerate os.add_dll_directory being missing
            # (as it is on non-Windows platforms at runtime) and return None.
            assert _setup_cuda_dll_paths() is None
            # The attribute should still be absent -- the function must not
            # accidentally install it as a side effect.
            assert not hasattr(os, "add_dll_directory")
        finally:
            if had_attr:
                os.add_dll_directory = original

    def test_adds_cuda_path_bin(self, tmp_path):
        from cyllama.utils.platform import _setup_cuda_dll_paths

        cuda_bin = tmp_path / "bin"
        cuda_bin.mkdir()

        mock_add = mock.MagicMock()
        with (
            mock.patch.dict(
                os.environ,
                {"CUDA_PATH": str(tmp_path), "CUDA_HOME": ""},
                clear=False,
            ),
            mock.patch.object(os, "add_dll_directory", mock_add, create=True),
            mock.patch("shutil.which", return_value=None),
        ):
            _setup_cuda_dll_paths()
            mock_add.assert_any_call(str(cuda_bin))

    def test_adds_nvcc_directory(self, tmp_path):
        from cyllama.utils.platform import _setup_cuda_dll_paths

        nvcc_dir = tmp_path / "bin"
        nvcc_dir.mkdir()
        nvcc_path = nvcc_dir / "nvcc"

        mock_add = mock.MagicMock()
        with (
            mock.patch.dict(
                os.environ,
                {"CUDA_PATH": "", "CUDA_HOME": ""},
                clear=False,
            ),
            mock.patch.object(os, "add_dll_directory", mock_add, create=True),
            mock.patch("shutil.which", return_value=str(nvcc_path)),
        ):
            _setup_cuda_dll_paths()
            mock_add.assert_any_call(str(nvcc_dir))

    def test_deduplicates_paths(self, tmp_path):
        from cyllama.utils.platform import _setup_cuda_dll_paths

        cuda_bin = tmp_path / "bin"
        cuda_bin.mkdir()

        mock_add = mock.MagicMock()
        with (
            mock.patch.dict(
                os.environ,
                {
                    "CUDA_PATH": str(tmp_path),
                    "CUDA_HOME": str(tmp_path),
                },
                clear=False,
            ),
            mock.patch.object(os, "add_dll_directory", mock_add, create=True),
            mock.patch("shutil.which", return_value=None),
        ):
            _setup_cuda_dll_paths()
            # Same bin dir from both env vars, should only be added once
            bin_calls = [c for c in mock_add.call_args_list if c == mock.call(str(cuda_bin))]
            assert len(bin_calls) == 1

    def test_standard_install_newest_first(self, tmp_path):
        from cyllama.utils.platform import _setup_cuda_dll_paths

        # Create fake CUDA install dirs
        cuda_dir = tmp_path / "NVIDIA GPU Computing Toolkit" / "CUDA"
        for ver in ("v11.8", "v12.4"):
            (cuda_dir / ver / "bin").mkdir(parents=True)

        added_dirs: list[str] = []

        def track_add(p: str) -> None:
            added_dirs.append(p)

        with (
            mock.patch.dict(
                os.environ,
                {
                    "CUDA_PATH": "",
                    "CUDA_HOME": "",
                    "ProgramFiles": str(tmp_path),
                },
                clear=False,
            ),
            mock.patch("shutil.which", return_value=None),
            mock.patch.object(os, "add_dll_directory", track_add, create=True),
        ):
            _setup_cuda_dll_paths()

        # v12.4 should come before v11.8
        v12_indices = [i for i, d in enumerate(added_dirs) if "v12.4" in d]
        v11_indices = [i for i, d in enumerate(added_dirs) if "v11.8" in d]
        assert v12_indices and v11_indices, f"Expected both versions in {added_dirs}"
        assert v12_indices[0] < v11_indices[0]
