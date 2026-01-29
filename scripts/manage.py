#!/usr/bin/env python3

"""manage.py: cross-platform cyllama build manager.

It only uses python stdlib modules to do the following:

- Dependency download, build, install
- Module compilation
- Wheel building
- Alternative frontend to Makefile
- Downloads/build a local version python for testing
- Multi-backend GPU support (Metal, CUDA, Vulkan, SYCL, HIP/ROCm, OpenCL)
- General Shell ops

models:
    CustomFormatter(logging.Formatter)
    MetaCommander(type)
    WheelFile(dataclass)
    ShellCmd
        Project
        AbstractBuilder
            Builder
                LlamaCppBuilder
                LlamaCppPythonBuilder
                WhisperCppBuilder
                StableDiffusionCppBuilder
        WheelBuilder
        Application(meta=MetaCommander)


It has an argparse-based cli api:

usage: manage.py [-h] [-v]  ...

cyllama build manager

    build        build application (with backend options)
    setup        setup prerequisites
    test         test modules
    wheel        build wheels
    clean        clean detritus
    info         show version info for dependencies
    download     download models (llama, whisper)
    bins         build llama.cpp CLI binaries
    bench        run performance benchmark (prefill/decode speed)
    profile      profile cyllama operations using cProfile

Backend support (via build command flags or environment variables):
    --metal, -m       Enable Metal backend (macOS)
    --cuda, -c        Enable CUDA backend (NVIDIA GPUs)
    --vulkan, -V      Enable Vulkan backend (cross-platform)
    --sycl, -y        Enable SYCL backend (Intel GPUs)
    --hip, -H         Enable HIP/ROCm backend (AMD GPUs)
    --opencl, -o      Enable OpenCL backend
    --cpu-only, -C    Disable all GPU backends

Environment variables:
    GGML_METAL=1      Enable Metal backend (llama.cpp, whisper.cpp)
    GGML_CUDA=1       Enable CUDA backend
    GGML_VULKAN=1     Enable Vulkan backend
    GGML_SYCL=1       Enable SYCL backend
    GGML_HIP=1        Enable HIP/ROCm backend
    GGML_OPENCL=1     Enable OpenCL backend
    SD_METAL=0        Disable Metal backend for stable-diffusion.cpp (default ON on macOS)
"""

import argparse
import logging
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import zipfile
from fnmatch import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union, Callable, NoReturn
from urllib.request import urlretrieve

__version__ = "0.1.1"

# ----------------------------------------------------------------------------
# type aliases

Pathlike = Union[str, Path]
MatchFn = Callable[[Path], bool]
ActionFn = Callable[[Path], None]

# ----------------------------------------------------------------------------
# env helpers


def getenv(key: str, default: bool = False) -> bool:
    """Convert '0','1' env values to bool {True, False}

    Args:
        key: Environment variable name
        default: Default value if not set

    Returns:
        Boolean value from environment variable

    Raises:
        ValueError: If environment variable value is not a valid integer
    """
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return bool(int(value))
    except ValueError:
        logging.getLogger(__name__).warning(
            f"Invalid boolean value for {key}: {value}, using default {default}"
        )
        return default


def setenv(key: str, default: str) -> str:
    """get environ variable if it is exists else set default"""
    if key in os.environ:
        return os.getenv(key, default)
    else:
        os.environ[key] = default
        return default


# ----------------------------------------------------------------------------
# constants

PYTHON = sys.executable
PLATFORM = platform.system()
ARCH = platform.machine()
PY_VER_MINOR = sys.version_info.minor

STABLE_BUILD = getenv("STABLE_BUILD", True)
if STABLE_BUILD:
    # known to build and work without errors, 100% tests pass
    LLAMACPP_VERSION = "b7823"
    WHISPERCPP_VERSION = "v1.8.3"
    SDCPP_VERSION = "master-487-43e829f"
    SQLITEVECTOR_VERSION = "0.9.52"
else:
    # experimental bleeding-edge builds ` = ""` means get latest
    LLAMACPP_VERSION = "b7823"
    WHISPERCPP_VERSION = "v1.8.3"
    SDCPP_VERSION = "master-487-43e829f"
    SQLITEVECTOR_VERSION = "0.9.52"
if PLATFORM == "Darwin":
    MACOSX_DEPLOYMENT_TARGET = setenv("MACOSX_DEPLOYMENT_TARGET", "12.6")
DEBUG = getenv("DEBUG", default=True)
COLOR = getenv("COLOR", default=True)

# ----------------------------------------------------------------------------
# logging config


class CustomFormatter(logging.Formatter):
    """custom logging formatting class"""

    white = "\x1b[97;20m"
    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    cyan = "\x1b[36;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    fmt = "%(asctime)s - {}%(levelname)s{} - %(name)s.%(funcName)s - %(message)s"

    FORMATS = {
        logging.DEBUG: fmt.format(grey, reset),
        logging.INFO: fmt.format(green, reset),
        logging.WARNING: fmt.format(yellow, reset),
        logging.ERROR: fmt.format(red, reset),
        logging.CRITICAL: fmt.format(bold_red, reset),
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%H:%M:%S")
        return formatter.format(record)


handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter())
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO, handlers=[handler])

# ----------------------------------------------------------------------------
# utility classes


class ShellCmd:
    """Provides platform agnostic file/folder handling."""

    log: logging.Logger

    def cmd(self, shellcmd: str, cwd: Pathlike = ".") -> None:
        """Run shell command within working directory

        WARNING: Uses shell=True for convenience. Only call with trusted input.

        Args:
            shellcmd: Shell command to execute (must be trusted input)
            cwd: Working directory for command execution

        Raises:
            SystemExit: If command fails
        """
        # Resolve and validate cwd path
        cwd_path = Path(cwd).resolve()

        self.log.info(shellcmd)
        try:
            subprocess.check_call(shellcmd, shell=True, cwd=str(cwd_path))
        except subprocess.CalledProcessError:
            self.log.critical("", exc_info=True)
            sys.exit(1)

    def download(
        self,
        url: str,
        tofolder: Optional[Pathlike] = None,
        max_size: int = 1024 * 1024 * 100,
    ) -> Pathlike:
        """Download a file from a url to an optional folder

        Args:
            url: URL to download from (must be http:// or https://)
            tofolder: Optional destination folder
            max_size: Maximum file size in bytes (default: 100MB)

        Returns:
            Path to downloaded file

        Raises:
            ValueError: If URL scheme is invalid, filename is unsafe, or file exceeds size limit
        """
        # Validate URL scheme
        if not url.startswith(("https://", "http://")):
            raise ValueError(f"Unsupported URL scheme: {url}")

        # Sanitize basename to prevent path traversal
        basename = os.path.basename(url)
        if ".." in basename or basename.startswith("/"):
            raise ValueError(f"Invalid filename in URL: {url}")

        _path = Path(basename)
        if tofolder:
            _path = Path(tofolder).resolve().joinpath(_path)
            if _path.exists():
                return _path

        self.log.info(f"Downloading {url} to {_path}")
        filename, _ = urlretrieve(url, filename=_path)

        # Check file size
        if _path.stat().st_size > max_size:
            _path.unlink()
            raise ValueError(
                f"Downloaded file exceeds size limit: {_path.stat().st_size} > {max_size}"
            )

        return Path(filename)

    def extract(self, archive: Pathlike, tofolder: Pathlike = ".") -> None:
        """Extract archive with path traversal protection

        Args:
            archive: Path to archive file
            tofolder: Destination folder for extraction

        Raises:
            ValueError: If archive contains files with path traversal attempts
            TypeError: If archive format is not supported
        """
        tofolder_resolved = Path(tofolder).resolve()

        def safe_extract_tar(members, dest):
            """Validate tar members before extraction"""
            for member in members:
                member_path = (dest / member.name).resolve()
                if not str(member_path).startswith(str(dest)):
                    raise ValueError(f"Attempted path traversal in tar: {member.name}")
            return members

        if tarfile.is_tarfile(archive):
            with tarfile.open(archive) as tar:
                safe_members = safe_extract_tar(tar.getmembers(), tofolder_resolved)
                tar.extractall(tofolder_resolved, members=safe_members)
        elif zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as zip_file:
                # Validate all zip members before extraction
                for info in zip_file.infolist():
                    extracted_path = (tofolder_resolved / info.filename).resolve()
                    if not str(extracted_path).startswith(str(tofolder_resolved)):
                        raise ValueError(
                            f"Attempted path traversal in zip: {info.filename}"
                        )
                zip_file.extractall(tofolder_resolved)
        else:
            raise TypeError("cannot extract from this file.")

    def fail(self, msg: str, *args: object) -> NoReturn:
        """exits the program with an error msg."""
        self.log.critical(msg, *args)
        sys.exit(1)

    def git_clone(
        self,
        url: str,
        branch: Optional[str] = None,
        directory: Optional[Pathlike] = None,
        recurse: bool = False,
        cwd: Pathlike = ".",
    ) -> None:
        """git clone a repository source tree from a url"""
        _cmds = ["git clone --depth 1"]
        if branch:
            _cmds.append(f"--branch {branch}")
        if recurse:
            _cmds.append("--recurse-submodules --shallow-submodules")
        _cmds.append(url)
        if directory:
            _cmds.append(str(directory))
        self.cmd(" ".join(_cmds), cwd=cwd)

    def getenv(self, key: str, default: bool = False) -> bool:
        """convert '0','1' env values to bool {True, False}"""
        self.log.info("checking env variable: %s", key)
        return bool(int(os.getenv(key, default)))

    def chdir(self, path: Pathlike) -> None:
        """Change current workding directory to path"""
        self.log.info("changing working dir to: %s", path)
        os.chdir(path)

    def chmod(self, path: Pathlike, perm: int = 0o777) -> None:
        """Change permission of file"""
        self.log.info("change permission of %s to %s", path, perm)
        os.chmod(path, perm)

    def get(
        self, shellcmd: Union[str, list[str]], cwd: Pathlike = ".", shell: bool = False
    ) -> str:
        """get output of shellcmd"""
        cmd_list: Union[str, list[str]]
        if not shell:
            if isinstance(shellcmd, str):
                cmd_list = shellcmd.split()
            else:
                cmd_list = shellcmd
        else:
            cmd_list = shellcmd
        return subprocess.check_output(
            cmd_list, encoding="utf8", shell=shell, cwd=str(cwd)
        ).strip()

    def makedirs(self, path: Pathlike, mode: int = 511, exist_ok: bool = True) -> None:
        """Recursive directory creation function"""
        self.log.info("making directory: %s", path)
        os.makedirs(path, mode, exist_ok)

    def move(self, src: Pathlike, dst: Pathlike) -> None:
        """Move from src path to dst path."""
        self.log.info("move path %s to %s", src, dst)
        shutil.move(src, dst)

    def copy(self, src: Pathlike, dst: Pathlike) -> None:
        """copy file or folders -- tries to be behave like `cp -rf`"""
        self.log.info("copy %s to %s", src, dst)
        src, dst = Path(src), Path(dst)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    def remove(self, path: Pathlike, silent: bool = False) -> None:
        """Remove file or folder."""

        # handle windows error on read-only files
        def remove_readonly(func, path, exc_info):
            "Clear the readonly bit and reattempt the removal"
            if PY_VER_MINOR < 11:
                if func not in (os.unlink, os.rmdir) or exc_info[1].winerror != 5:
                    raise exc_info[1]
            else:
                if func not in (os.unlink, os.rmdir) or exc_info.winerror != 5:
                    raise exc_info
            os.chmod(path, stat.S_IWRITE)
            func(path)

        path = Path(path)
        if path.is_dir():
            if not silent:
                self.log.info("remove folder: %s", path)
            if PY_VER_MINOR < 11:
                shutil.rmtree(path, ignore_errors=not DEBUG, onerror=remove_readonly)
            else:
                shutil.rmtree(path, ignore_errors=not DEBUG, onexc=remove_readonly)
        else:
            if not silent:
                self.log.info("remove file: %s", path)
            try:
                path.unlink()
            except FileNotFoundError:
                if not silent:
                    self.log.warning("file not found: %s", path)

    def walk(
        self,
        root: Pathlike,
        match_func: MatchFn,
        action_func: ActionFn,
        skip_patterns: list[str],
    ) -> None:
        """general recursive walk from root path with match and action functions"""
        for root_, dirs, filenames in os.walk(root):
            _root = Path(root_)
            if skip_patterns:
                for skip_pat in skip_patterns:
                    if skip_pat in dirs:
                        dirs.remove(skip_pat)
            for _dir in dirs:
                current = _root / _dir
                if match_func(current):
                    action_func(current)

            for _file in filenames:
                current = _root / _file
                if match_func(current):
                    action_func(current)

    def glob_copy(
        self,
        src: Pathlike,
        dest: Pathlike,
        patterns: list[str],
    ) -> None:
        """copy glob patterns from src dir to destination dir"""

        src = Path(src)
        dest = Path(dest)

        if not src.exists():
            raise IOError(f"src dir '{src}' not found")

        if not dest.exists():
            dest.mkdir()

        for p in patterns:
            for f in src.glob(p):
                self.copy(f, dest)

    def glob_remove(
        self, root: Pathlike, patterns: list[str], skip_dirs: list[str]
    ) -> None:
        """applies recursive glob remove using a list of patterns"""

        def _match(entry: Path) -> bool:
            # return any(fnmatch(entry, p) for p in patterns)
            return any(fnmatch(entry.name, p) for p in patterns)

        def remove(entry: Path):
            self.remove(entry)

        self.walk(root, match_func=_match, action_func=remove, skip_patterns=skip_dirs)

    def pip_install(
        self,
        *pkgs: str,
        reqs: Optional[str] = None,
        upgrade: bool = False,
        pip: Optional[str] = None,
    ) -> None:
        """Install python packages using pip"""
        _cmds = []
        if pip:
            _cmds.append(pip)
        else:
            _cmds.append("pip3")
        _cmds.append("install")
        if reqs:
            _cmds.append(f"-r {reqs}")
        else:
            if upgrade:
                _cmds.append("--upgrade")
            _cmds.extend(pkgs)
        self.cmd(" ".join(_cmds))

    def apt_install(self, *pkgs: str, update: bool = False) -> None:
        """install debian packages using apt"""
        _cmds = []
        _cmds.append("sudo apt install")
        if update:
            _cmds.append("--upgrade")
        _cmds.extend(pkgs)
        self.cmd(" ".join(_cmds))

    def brew_install(self, *pkgs: str, update: bool = False) -> None:
        """install using homebrew"""
        _pkgs = " ".join(pkgs)
        if update:
            self.cmd("brew update")
        self.cmd(f"brew install {_pkgs}")

    def cmake_config(
        self,
        src_dir: Pathlike,
        build_dir: Pathlike,
        *scripts: str,
        **options: Union[str, bool, int],
    ) -> None:
        """activate cmake configuration / generation stage"""
        _cmds = [f"cmake -S {src_dir} -B {build_dir}"]
        if scripts:
            _cmds.append(" ".join(f"-C {path}" for path in scripts))
        if options:
            # Convert Python bools to CMake ON/OFF
            def cmake_value(v):
                if isinstance(v, bool):
                    return "ON" if v else "OFF"
                return v

            _cmds.append(
                " ".join(f"-D{k}={cmake_value(v)}" for k, v in options.items())
            )
        self.cmd(" ".join(_cmds))

    def cmake_build(self, build_dir: Pathlike, release: bool = False) -> None:
        """activate cmake build stage"""
        _cmd = f"cmake --build {build_dir}"
        if release:
            _cmd += " --config Release"
        self.cmd(_cmd)

    def cmake_build_targets(
        self, build_dir: Pathlike, targets: list[str], release: bool = False
    ) -> None:
        """build specific cmake targets"""
        _cmd = f"cmake --build {build_dir}"
        if release:
            _cmd += " --config Release"
        for target in targets:
            _cmd += f" --target {target}"
        self.cmd(_cmd)

    def cmake_install(
        self, build_dir: Pathlike, prefix: Optional[Pathlike] = None
    ) -> None:
        """activate cmake install stage"""
        _cmds = ["cmake --install", str(build_dir)]
        if prefix:
            _cmds.append(f"--prefix {str(prefix)}")
        self.cmd(" ".join(_cmds))


# ----------------------------------------------------------------------------
# main classes


class Project(ShellCmd):
    """Utility class to hold project directory structure"""

    cwd: Path
    build: Path
    src: Path
    thirdparty: Path
    install: Path
    dist: Path
    scripts: Path
    tests: Path
    wheels: Path
    lib: Path

    def __init__(self) -> None:
        self.cwd = Path.cwd()
        self.build = self.cwd / "build"
        # self.src = self.build / "repos"
        self.src = self.build
        self.thirdparty = self.cwd / "thirdparty"
        self.install = self.thirdparty
        self.dist = self.cwd / "dist"
        self.scripts = self.cwd / "scripts"
        self.tests = self.cwd / "tests"
        self.wheels = self.cwd / "wheels"
        self.lib = self.thirdparty / "llama.cpp" / "lib"

    def setup(self) -> None:
        """create main project directories"""
        # self.bin.mkdir(exist_ok=True)
        self.build.mkdir(exist_ok=True)
        self.src.mkdir(exist_ok=True)
        self.install.mkdir(exist_ok=True)

    def clean(self) -> None:
        """prepare project for a partial rebuild"""
        self.remove(self.build)
        self.remove(self.dist)

    def reset(self) -> None:
        """prepare project for a full rebuild"""
        self.clean()
        self.remove(self.install)


class AbstractBuilder(ShellCmd):
    """Abstract builder class with additional methods common to subclasses."""

    name: str
    version: str
    repo_url: str
    download_url_template: str
    libs_static: list[str]
    depends_on: list[type["Builder"]]

    def __init__(
        self, version: Optional[str] = None, project: Optional[Project] = None
    ):
        self.version = version or self.version
        self.project = project or Project()
        self.log = logging.getLogger(self.__class__.__name__)

    def __repr__(self):
        return f"<{self.__class__.__name__} '{self.name}-{self.version}'>"

    # def __iter__(self):
    #     for dependency in self.depends_on:
    #         yield dependency
    #         yield from iter(dependency)

    @property
    def ver(self) -> str:
        """short python version: 3.11"""
        return ".".join(self.version.split(".")[:2])

    @property
    def ver_major(self) -> str:
        """major compoent of semantic version: 3 in 3.11.7"""
        return self.version.split(".")[0]

    @property
    def ver_minor(self) -> str:
        """minor compoent of semantic version: 11 in 3.11.7"""
        return self.version.split(".")[1]

    @property
    def ver_patch(self) -> str:
        """patch compoent of semantic version: 7 in 3.11.7"""
        return self.version.split(".")[2]

    @property
    def ver_nodot(self) -> str:
        """concat major and minor version components: 311 in 3.11.7"""
        return self.ver.replace(".", "")

    @property
    def name_version(self) -> str:
        """return name-<fullversion>: e.g. Python-3.11.7"""
        return f"{self.name}-{self.version}"

    @property
    def name_ver(self) -> str:
        """return name.lower-<ver>: e.g. python3.11"""
        return f"{self.name.lower()}{self.ver}"

    @property
    def download_url(self) -> str:
        """return download url with version interpolated"""
        return self.download_url_template.format(ver=self.version)

    @property
    def repo_branch(self) -> str:
        """return repo branch"""
        return self.name.lower()

    @property
    def src_dir(self) -> Path:
        """return extracted source folder of build target"""
        return self.project.src / self.name

    @property
    def build_dir(self) -> Path:
        """return 'build' folder src dir of build target"""
        return self.src_dir / "build"

    @property
    def prefix(self) -> Path:
        """builder prefix path"""
        return self.project.install / self.name.lower()

    @property
    def bin(self) -> Path:
        """builder bin path"""
        return self.prefix / "bin"

    @property
    def include(self) -> Path:
        """builder include path"""
        return self.prefix / "include"

    @property
    def lib(self) -> Path:
        """builder lib path"""
        return self.prefix / "lib"

    @property
    def executable_name(self) -> str:
        """executable name of buld target"""
        name = self.name.lower()
        if PLATFORM == "Windows":
            name = f"{self.name}.exe"
        return name

    @property
    def executable(self) -> Path:
        """executable path of buld target"""
        return self.bin / self.executable_name

    @property
    def libname(self) -> str:
        """library name prefix"""
        return f"lib{self.name}"

    @property
    def staticlib_name(self) -> str:
        """static libname"""
        suffix = ".a"
        if PLATFORM == "Windows":
            suffix = ".lib"
        return f"{self.libname}{suffix}"

    @property
    def dylib_name(self) -> str:
        """dynamic link libname"""
        if PLATFORM == "Darwin":
            return f"{self.libname}.dylib"
        if PLATFORM == "Linux":
            return f"{self.libname}.so"
        if PLATFORM == "Windows":
            return f"{self.libname}.dll"
        return self.fail("platform not supported")

    @property
    def dylib_linkname(self) -> str:
        """symlink to dylib"""
        if PLATFORM == "Darwin":
            return f"{self.libname}.dylib"
        if PLATFORM == "Linux":
            return f"{self.libname}.so"
        return self.fail("platform not supported")

    @property
    def dylib(self) -> Path:
        """dylib path"""
        return self.lib / self.dylib_name

    @property
    def dylib_link(self) -> Path:
        """dylib link path"""
        return self.lib / self.dylib_linkname

    @property
    def staticlib(self) -> Path:
        """staticlib path"""
        return self.lib / self.staticlib_name

    def get_lib_path(self, build_dir: Path, subdir: str, name: str) -> Path:
        """Get platform-specific library path from build directory.

        On Windows, CMake puts libraries in Release/ subdirectory and uses
        name.lib format. On Unix, it's libname.a directly in the directory.

        Args:
            build_dir: The CMake build directory
            subdir: Subdirectory within build_dir (e.g., "common", "src")
            name: Library name without prefix/extension (e.g., "common", "llama")

        Returns:
            Path to the library file
        """
        base = build_dir / subdir

        if PLATFORM == "Windows":
            # Try Release/ subdirectory first (multi-config generators)
            release_path = base / "Release" / f"{name}.lib"
            if release_path.exists():
                return release_path
            # Fall back to direct path (single-config generators)
            direct_path = base / f"{name}.lib"
            if direct_path.exists():
                return direct_path
            # Return expected Release path for error messages
            return release_path
        else:
            # Unix: libname.a directly in directory
            return base / f"lib{name}.a"

    def copy_lib(self, build_dir: Path, subdir: str, name: str, dest: Path) -> bool:
        """Copy a library from build directory to destination.

        Args:
            build_dir: The CMake build directory
            subdir: Subdirectory within build_dir
            name: Library name without prefix/extension
            dest: Destination directory

        Returns:
            True if copied successfully, False otherwise
        """
        lib_path = self.get_lib_path(build_dir, subdir, name)
        if lib_path.exists():
            self.copy(lib_path, dest)
            self.log.info(f"Copied {lib_path.name} to {dest}")
            return True
        else:
            self.log.warning(f"Library not found: {lib_path}")
            return False

    def libs_static_exist(self) -> bool:
        """check if all built stati libs already exist"""
        return all((self.lib / lib).exists() for lib in self.libs_static)

    def pre_process(self) -> None:
        """override by subclass if needed"""

    def setup(self) -> None:
        """setup build environment"""

    def configure(self) -> None:
        """configure build"""

    def build(self, shared: bool = False) -> None:
        """build target"""

    def install(self) -> None:
        """install target"""

    def clean(self) -> None:
        """clean build"""

    def post_process(self) -> None:
        """override by subclass if needed"""

    def process(self) -> None:
        """main builder process"""
        self.pre_process()
        self.setup()
        self.configure()
        self.build()
        self.install()
        self.clean()
        self.post_process()


class Builder(AbstractBuilder):
    """concrete builder class"""

    def setup(self) -> None:
        """setup build environment"""
        self.log.info(f"update from {self.name} main repo")
        self.project.setup()
        if self.version:
            self.git_clone(
                self.repo_url, branch=self.version, recurse=True, cwd=self.project.src
            )
        else:
            self.git_clone(self.repo_url, recurse=True, cwd=self.project.src)


class LlamaCppBuilder(Builder):
    """build llama.cpp"""

    name: str = "llama.cpp"
    version: str = LLAMACPP_VERSION
    repo_url: str = "https://github.com/ggml-org/llama.cpp.git"
    libs_static: list[str] = [
        "libcommon.a",
        "libggml-base.a",
        "libggml-blas.a",
        "libggml-cpu.a",
        "libggml-metal.a",
        "libggml.a",
        "libllama.a",
        "libmtmd.a",
    ]

    def get_backend_cmake_options(self) -> dict:
        """Get CMake options based on backend environment variables."""
        options = {}

        # Read backend flags from environment (default Metal=1 on macOS, others=0)
        ggml_metal = getenv(
            "GGML_METAL", default=(True if PLATFORM == "Darwin" else False)
        )
        ggml_cuda = getenv("GGML_CUDA", default=False)
        ggml_vulkan = getenv("GGML_VULKAN", default=False)
        ggml_sycl = getenv("GGML_SYCL", default=False)
        ggml_hip = getenv("GGML_HIP", default=False)
        ggml_opencl = getenv("GGML_OPENCL", default=False)

        # Add CMake options for enabled backends
        if ggml_metal:
            options["GGML_METAL"] = "ON"
            self.log.info("✓ Enabling Metal backend")

        if ggml_cuda:
            options["GGML_CUDA"] = "ON"
            self.log.info("✓ Enabling CUDA backend")

        if ggml_vulkan:
            options["GGML_VULKAN"] = "ON"
            self.log.info("✓ Enabling Vulkan backend")

        if ggml_sycl:
            options["GGML_SYCL"] = "ON"
            self.log.info("✓ Enabling SYCL backend")

        if ggml_hip:
            options["GGML_HIP"] = "ON"
            self.log.info("✓ Enabling HIP/ROCm backend")

        if ggml_opencl:
            options["GGML_OPENCL"] = "ON"
            self.log.info("✓ Enabling OpenCL backend")

        return options

    def copy_backend_libs(self) -> None:
        """Copy backend-specific libraries based on enabled backends."""
        # Read backend flags from environment
        ggml_metal = getenv(
            "GGML_METAL", default=(True if PLATFORM == "Darwin" else False)
        )
        ggml_cuda = getenv("GGML_CUDA", default=False)
        ggml_vulkan = getenv("GGML_VULKAN", default=False)
        ggml_sycl = getenv("GGML_SYCL", default=False)
        ggml_hip = getenv("GGML_HIP", default=False)
        ggml_opencl = getenv("GGML_OPENCL", default=False)

        # Copy Metal backend libraries (macOS only)
        if ggml_metal:
            self.copy_lib(self.build_dir, "ggml/src/ggml-blas", "ggml-blas", self.lib)
            self.copy_lib(self.build_dir, "ggml/src/ggml-metal", "ggml-metal", self.lib)

        # Copy CUDA backend library
        if ggml_cuda:
            self.copy_lib(self.build_dir, "ggml/src/ggml-cuda", "ggml-cuda", self.lib)

        # Copy Vulkan backend library
        if ggml_vulkan:
            self.copy_lib(
                self.build_dir, "ggml/src/ggml-vulkan", "ggml-vulkan", self.lib
            )

        # Copy SYCL backend library
        if ggml_sycl:
            self.copy_lib(self.build_dir, "ggml/src/ggml-sycl", "ggml-sycl", self.lib)

        # Copy HIP backend library
        if ggml_hip:
            self.copy_lib(self.build_dir, "ggml/src/ggml-hip", "ggml-hip", self.lib)

        # Copy OpenCL backend library
        if ggml_opencl:
            self.copy_lib(
                self.build_dir, "ggml/src/ggml-opencl", "ggml-opencl", self.lib
            )

    def build(self, shared: bool = False) -> None:
        """main build function"""
        if not self.src_dir.exists():
            self.setup()
        self.log.info(f"building {self.name}")
        self.prefix.mkdir(exist_ok=True)
        self.include.mkdir(exist_ok=True)
        self.glob_copy(self.src_dir / "common", self.include, patterns=["*.h", "*.hpp"])
        self.glob_copy(
            self.src_dir / "ggml" / "include", self.include, patterns=["*.h"]
        )
        # Copy main llama.h header from include/ directory
        self.glob_copy(self.src_dir / "include", self.include, patterns=["*.h"])
        # Copy nlohmann JSON headers (required by json-partial.h)
        nlohmann_include = self.include / "nlohmann"
        nlohmann_include.mkdir(exist_ok=True)
        self.glob_copy(
            self.src_dir / "vendor" / "nlohmann", nlohmann_include, patterns=["*.hpp"]
        )
        # Copy mtmd (multimodal) headers
        self.glob_copy(self.src_dir / "tools" / "mtmd", self.include, patterns=["*.h"])

        # Get backend-specific CMake options
        backend_options = self.get_backend_cmake_options()

        self.cmake_config(
            src_dir=self.src_dir,
            build_dir=self.build_dir,
            BUILD_SHARED_LIBS=shared,
            CMAKE_POSITION_INDEPENDENT_CODE=True,
            CMAKE_CXX_VISIBILITY_PRESET="hidden",
            CMAKE_C_VISIBILITY_PRESET="hidden",
            CMAKE_VISIBILITY_INLINES_HIDDEN=True,
            LLAMA_CURL=False,
            LLAMA_HTTPLIB=False,  # Disable httplib to avoid linking issues
            LLAMA_BUILD_SERVER=False,  # Server requires httplib
            LLAMA_BUILD_TESTS=False,  # Tests require httplib
            LLAMA_BUILD_EXAMPLES=False,  # Don't need examples
            **backend_options,
        )
        # Build specific targets to avoid httplib-dependent tools like llama-run
        # We need: llama, ggml, common, mtmd
        self.cmake_build_targets(
            build_dir=self.build_dir, targets=["llama", "common", "mtmd"], release=True
        )

        # Manually copy required libraries instead of cmake install (which tries to install all components)
        self.lib.mkdir(parents=True, exist_ok=True)

        # Copy core libraries from build directory (platform-aware)
        self.copy_lib(self.build_dir, "common", "common", self.lib)
        self.copy_lib(self.build_dir, "src", "llama", self.lib)
        self.copy_lib(self.build_dir, "ggml/src", "ggml", self.lib)
        self.copy_lib(self.build_dir, "ggml/src", "ggml-base", self.lib)
        self.copy_lib(self.build_dir, "ggml/src", "ggml-cpu", self.lib)
        self.copy_lib(self.build_dir, "tools/mtmd", "mtmd", self.lib)

        # Copy backend-specific libraries
        self.copy_backend_libs()

        # self.move(self.prefix / "bin", self.project.bin)


class WhisperCppBuilder(Builder):
    """build whisper.cpp"""

    name: str = "whisper.cpp"
    version: str = WHISPERCPP_VERSION
    repo_url: str = "https://github.com/ggml-org/whisper.cpp"
    libs_static: list[str] = [
        "libcommon.a",
        "libwhisper.a",
        "libggml.a",
    ]

    def get_backend_cmake_options(self) -> dict:
        """Get CMake options based on backend environment variables.

        whisper.cpp uses GGML_* flags (same as llama.cpp).
        """
        options = {}

        # Read backend flags from environment (default Metal=1 on macOS, others=0)
        ggml_metal = getenv(
            "GGML_METAL", default=(True if PLATFORM == "Darwin" else False)
        )
        ggml_cuda = getenv("GGML_CUDA", default=False)
        ggml_vulkan = getenv("GGML_VULKAN", default=False)
        ggml_sycl = getenv("GGML_SYCL", default=False)
        ggml_hip = getenv("GGML_HIP", default=False)
        ggml_opencl = getenv("GGML_OPENCL", default=False)

        # Explicitly disable Metal on non-macOS
        if PLATFORM != "Darwin":
            options["GGML_METAL"] = "OFF"

        if ggml_metal and PLATFORM == "Darwin":
            options["GGML_METAL"] = "ON"
            self.log.info("Enabling Metal backend for whisper.cpp")

        if ggml_cuda:
            options["GGML_CUDA"] = "ON"
            self.log.info("Enabling CUDA backend for whisper.cpp")

        if ggml_vulkan:
            options["GGML_VULKAN"] = "ON"
            self.log.info("Enabling Vulkan backend for whisper.cpp")

        if ggml_sycl:
            options["GGML_SYCL"] = "ON"
            self.log.info("Enabling SYCL backend for whisper.cpp")

        if ggml_hip:
            options["GGML_HIP"] = "ON"
            self.log.info("Enabling HIP/ROCm backend for whisper.cpp")

        if ggml_opencl:
            options["GGML_OPENCL"] = "ON"
            self.log.info("Enabling OpenCL backend for whisper.cpp")

        return options

    def build(self, shared: bool = False) -> None:
        """whisper.cpp main build function"""
        if not self.src_dir.exists():
            self.setup()
        self.log.info(f"building {self.name}")
        self.prefix.mkdir(exist_ok=True)
        self.include.mkdir(exist_ok=True)
        self.glob_copy(
            self.src_dir / "examples", self.include, patterns=["*.h", "*.hpp"]
        )

        # Get backend options
        backend_options = self.get_backend_cmake_options()

        self.cmake_config(
            src_dir=self.src_dir,
            build_dir=self.build_dir,
            BUILD_SHARED_LIBS=shared,
            CMAKE_POSITION_INDEPENDENT_CODE=True,
            CMAKE_CXX_VISIBILITY_PRESET="hidden",
            CMAKE_C_VISIBILITY_PRESET="hidden",
            CMAKE_VISIBILITY_INLINES_HIDDEN=True,
            **backend_options,
        )
        self.cmake_build(build_dir=self.build_dir, release=True)
        self.cmake_install(build_dir=self.build_dir, prefix=self.prefix)
        self.copy_lib(self.build_dir, "examples", "common", self.lib)
        # self.glob_copy(self.build_dir / "bin", self.bin, patterns=["*"])


class StableDiffusionCppBuilder(Builder):
    """build stable-diffusion.cpp"""

    name: str = "stable-diffusion.cpp"
    version: str = SDCPP_VERSION
    repo_url: str = "https://github.com/leejet/stable-diffusion.cpp.git"
    libs_static: list[str] = [
        "libstable-diffusion.a",
    ]

    def get_backend_cmake_options(self) -> dict:
        """Get CMake options based on backend environment variables.

        stable-diffusion.cpp uses SD_* flags (not GGML_*).
        SD_METAL defaults to ON on macOS.
        """
        options = {}

        # Read backend flags from environment (default Metal=1 on macOS, others=0)
        sd_metal = getenv(
            "SD_METAL", default=(True if PLATFORM == "Darwin" else False)
        )
        ggml_cuda = getenv("GGML_CUDA", default=False)
        ggml_vulkan = getenv("GGML_VULKAN", default=False)
        ggml_sycl = getenv("GGML_SYCL", default=False)
        ggml_hip = getenv("GGML_HIP", default=False)
        ggml_opencl = getenv("GGML_OPENCL", default=False)

        if sd_metal and PLATFORM == "Darwin":
            options["SD_METAL"] = "ON"
            self.log.info("Enabling Metal backend for stable-diffusion.cpp")

        if ggml_cuda:
            options["SD_CUDA"] = "ON"
            self.log.info("Enabling CUDA backend for stable-diffusion.cpp")

        if ggml_vulkan:
            options["SD_VULKAN"] = "ON"
            self.log.info("Enabling Vulkan backend for stable-diffusion.cpp")

        if ggml_sycl:
            options["SD_SYCL"] = "ON"
            self.log.info("Enabling SYCL backend for stable-diffusion.cpp")

        if ggml_hip:
            options["SD_HIPBLAS"] = "ON"
            self.log.info("Enabling HIP/ROCm backend for stable-diffusion.cpp")

        if ggml_opencl:
            options["SD_OPENCL"] = "ON"
            self.log.info("Enabling OpenCL backend for stable-diffusion.cpp")

        return options

    def build(self, shared: bool = False) -> None:
        """stable-diffusion.cpp main build function"""
        if not self.src_dir.exists():
            self.setup()
        self.log.info(f"building {self.name}")
        self.prefix.mkdir(exist_ok=True)
        self.include.mkdir(exist_ok=True)
        self.glob_copy(self.src_dir, self.include, patterns=["*.h", "*.hpp"])
        # Copy stb headers for zero-dependency image I/O
        stb_src = self.src_dir / "thirdparty"
        if stb_src.exists():
            for stb_file in ["stb_image.h", "stb_image_write.h", "stb_image_resize.h"]:
                stb_path = stb_src / stb_file
                if stb_path.exists():
                    self.copy(stb_path, self.include)
                    self.log.info(f"Copied {stb_file} to include directory")

        # Get backend options
        backend_options = self.get_backend_cmake_options()

        self.cmake_config(
            src_dir=self.src_dir,
            build_dir=self.build_dir,
            BUILD_SHARED_LIBS=shared,
            CMAKE_POSITION_INDEPENDENT_CODE=True,
            CMAKE_CXX_VISIBILITY_PRESET="hidden",
            CMAKE_C_VISIBILITY_PRESET="hidden",
            CMAKE_VISIBILITY_INLINES_HIDDEN=True,
            **backend_options,
        )
        self.cmake_build(build_dir=self.build_dir, release=True)
        self.cmake_install(build_dir=self.build_dir, prefix=self.prefix)
        self.copy_lib(self.build_dir, ".", "stable-diffusion", self.lib)


class SqliteVectorBuilder(Builder):
    """build sqlite-vector extension"""

    name: str = "sqlite-vector"
    version: str = SQLITEVECTOR_VERSION
    repo_url: str = "https://github.com/sqliteai/sqlite-vector.git"
    libs_static: list[str] = []  # sqlite-vector produces a dynamic library

    @property
    def extension_name(self) -> str:
        """Get platform-specific extension name"""
        if PLATFORM == "Darwin":
            return "vector.dylib"
        elif PLATFORM == "Windows":
            return "vector.dll"
        else:
            return "vector.so"

    @property
    def package_dest(self) -> Path:
        """Destination directory in the package for runtime extension"""
        return self.project.cwd / "src" / "cyllama" / "rag"

    def build(self, shared: bool = True) -> None:
        """sqlite-vector main build function using make"""
        if not self.src_dir.exists():
            self.setup()
        self.log.info(f"building {self.name}")

        # Ensure destination directory exists
        self.package_dest.mkdir(parents=True, exist_ok=True)

        # Clean any previous build
        self.cmd("make clean", cwd=self.src_dir)

        # Build the extension using make
        self.cmd("make extension", cwd=self.src_dir)

        # Copy the extension to package directory (for runtime use)
        dist_dir = self.src_dir / "dist"
        ext_path = dist_dir / self.extension_name
        if ext_path.exists():
            self.copy(ext_path, self.package_dest)
            self.log.info(f"Copied {self.extension_name} to {self.package_dest}")
        else:
            self.log.warning(f"Extension not found: {ext_path}")


# ----------------------------------------------------------------------------
# wheel_builder


@dataclass
class WheelFilename:
    """Wheel filename dataclass with parser.

    credits:
        wheel parsing code is derived from
        from https://github.com/wheelodex/wheel-filename
        Copyright (c) 2020-2022 John Thorvald Wodder II

    This version uses dataclasses instead of NamedTuples in the original
    and packages the parsing function and the regex patterns in the
    class itself.
    """

    PYTHON_TAG_RGX = r"[\w\d]+"
    ABI_TAG_RGX = r"[\w\d]+"
    PLATFORM_TAG_RGX = r"[\w\d]+"

    WHEEL_FILENAME_PATTERN = re.compile(
        r"(?P<project>[A-Za-z0-9](?:[A-Za-z0-9._]*[A-Za-z0-9])?)"
        r"-(?P<version>[A-Za-z0-9_.!+]+)"
        r"(?:-(?P<build>[0-9][\w\d.]*))?"
        r"-(?P<python_tags>{0}(?:\.{0})*)"
        r"-(?P<abi_tags>{1}(?:\.{1})*)"
        r"-(?P<platform_tags>{2}(?:\.{2})*)"
        r"\.[Ww][Hh][Ll]".format(PYTHON_TAG_RGX, ABI_TAG_RGX, PLATFORM_TAG_RGX)
    )

    project: str
    version: str
    build: Optional[str]
    python_tags: List[str]
    abi_tags: List[str]
    platform_tags: List[str]

    def __str__(self) -> str:
        if self.build:
            fmt = "{0.project}-{0.version}-{0.build}-{1}-{2}-{3}.whl"
        else:
            fmt = "{0.project}-{0.version}-{1}-{2}-{3}.whl"
        return fmt.format(
            self,
            ".".join(self.python_tags),
            ".".join(self.abi_tags),
            ".".join(self.platform_tags),
        )

    @classmethod
    def from_path(cls, path: Pathlike) -> "WheelFilename":
        """Parse a wheel filename into its components"""
        basename = Path(path).name
        m = cls.WHEEL_FILENAME_PATTERN.fullmatch(basename)
        if not m:
            raise TypeError("incorrect wheel name")
        return cls(
            project=m.group("project"),
            version=m.group("version"),
            build=m.group("build"),
            python_tags=m.group("python_tags").split("."),
            abi_tags=m.group("abi_tags").split("."),
            platform_tags=m.group("platform_tags").split("."),
        )


class WheelBuilder(ShellCmd):
    """cyllama wheel builder

    Automates wheel building and handle special cases
    when building cyllama locally and on github actions,
    especially whenc considering the number of different products given
    build-variants * platforms * architectures:
        {dynamic, static} * {macos, linux} * {x86_64, arm64|aarch64}
    """

    universal: bool
    project: Project

    def __init__(self, universal: bool = False) -> None:
        self.universal = universal
        self.project = Project()
        self.log = logging.getLogger(self.__class__.__name__)

    def get_min_osx_ver(self) -> str:
        """set MACOSX_DEPLOYMENT_TARGET

        credits: cibuildwheel
        ref: https://github.com/pypa/cibuildwheel/blob/main/cibuildwheel/macos.py
        thanks: @henryiii
        post: https://github.com/pypa/wheel/issues/573

        For arm64, the minimal deployment target is 11.0.
        On x86_64 (or universal2), use 10.9 as a default.
        """
        min_osx_ver = "10.9"
        if self.is_macos_arm64 and not self.universal:
            min_osx_ver = "11.0"
        os.environ["MACOSX_DEPLOYMENT_TARGET"] = min_osx_ver
        return min_osx_ver

    @property
    def is_static(self) -> bool:
        return self.getenv("STATIC")

    @property
    def is_macos_arm64(self) -> bool:
        return PLATFORM == "Darwin" and ARCH == "arm64"

    @property
    def is_macos_x86_64(self) -> bool:
        return PLATFORM == "Darwin" and ARCH == "x86_64"

    @property
    def is_linux_x86_64(self) -> bool:
        return PLATFORM == "Linux" and ARCH == "x86_64"

    @property
    def is_linux_aarch64(self) -> bool:
        return PLATFORM == "Linux" and ARCH == "aarch64"

    def clean(self) -> None:
        if self.project.build.exists():
            shutil.rmtree(self.project.build, ignore_errors=True)
        if self.project.dist.exists():
            shutil.rmtree(self.project.dist)

    def reset(self) -> None:
        self.clean()
        if self.project.wheels.exists():
            shutil.rmtree(self.project.wheels)

    def check(self) -> None:
        have_wheels = bool(self.project.wheels.glob("*.whl"))
        if not have_wheels:
            self.fail("no wheels created")

    def ensure_wheels_dir(self) -> None:
        """Ensure wheels directory exists"""
        if not self.project.wheels.exists():
            self.project.wheels.mkdir()

    def build_wheel(self, static: bool = False, override: bool = True) -> None:
        assert PY_VER_MINOR >= 8, "only supporting python >= 3.8"

        # Build wheel using scikit-build-core via uv
        _cmd = "uv build --wheel"

        if PLATFORM == "Darwin":
            ver = self.get_min_osx_ver()
            if self.universal:
                prefix = (
                    f"ARCHFLAGS='-arch arm64 -arch x86_64' "
                    f"_PYTHON_HOST_PLATFORM='macosx-{ver}-universal2' "
                )
                _cmd = prefix + _cmd

        self.cmd(_cmd)

    def test_wheels(self) -> None:
        venv = self.project.wheels / "venv"
        if venv.exists():
            shutil.rmtree(venv)

        for wheel in self.project.wheels.glob("*.whl"):
            self.cmd("virtualenv venv", cwd=self.project.wheels)
            if PLATFORM in ["Linux", "Darwin"]:
                vpy = venv / "bin" / "python"
                vpip = venv / "bin" / "pip"
            elif PLATFORM == "Windows":
                vpy = venv / "Scripts" / "python"
                vpip = venv / "Scripts" / "pip"
            else:
                self.fail("platform not supported")

            self.cmd(f"{vpip} install {wheel}")
            if "static" in str(wheel):
                target = "static"
                imported = "cyllama"
                self.log.info("static variant test")
            else:
                target = "dynamic"
                imported = "interp"
                self.log.info("dynamic variant test")
            val = self.get(
                f'{vpy} -c "from cyllama import {imported};print(len(dir({imported})))"',
                shell=True,
                cwd=self.project.wheels,
            )
            self.log.info(f"cyllama.{imported} # objects: {val}")
            assert val, f"cyllama {target} wheel test: FAILED"
            self.log.info(f"cyllama {target} wheel test: OK")
            if venv.exists():
                shutil.rmtree(venv)

    def build_dynamic_wheel(self) -> None:
        self.log.info("building dynamic build wheel")
        self.clean()
        self.ensure_wheels_dir()
        self.build_wheel()
        src = self.project.dist
        dst = self.project.wheels
        lib = self.project.lib
        if PLATFORM == "Darwin":
            self.cmd(f"delocate-wheel -v --wheel-dir {dst} {src}/*.whl")
        elif PLATFORM == "Linux":
            self.cmd(
                f"auditwheel repair --plat linux_{ARCH} --wheel-dir {dst} {src}/*.whl"
            )
        elif PLATFORM == "Windows":
            for whl in self.project.dist.glob("*.whl"):
                self.cmd(f"delvewheel repair --add-path {lib} --wheel-dir {dst} {whl}")
        else:
            raise self.fail("platform not supported")

    def build_static_wheel(self) -> None:
        self.log.info("building static build wheel")
        self.clean()
        self.ensure_wheels_dir()
        self.build_wheel(static=True)
        for wheel in self.project.dist.glob("*.whl"):
            w = WheelFilename.from_path(wheel)
            w.project = "cyllama-static"
            renamed_wheel = str(w)
            os.rename(wheel, renamed_wheel)
            shutil.move(renamed_wheel, self.project.wheels)

    def build(self) -> None:
        if self.is_static:
            self.build_static_wheel()
        else:
            self.build_dynamic_wheel()
        self.check()
        self.clean()

    def release(self) -> None:
        self.reset()
        self.build_dynamic_wheel()
        self.build_static_wheel()
        self.check()
        self.clean()


# ----------------------------------------------------------------------------
# argdeclare


# option decorator
def option(*args, **kwds):
    def _decorator(func):
        _option = (args, kwds)
        if hasattr(func, "options"):
            func.options.append(_option)
        else:
            func.options = [_option]
        return func

    return _decorator


# bool option decorator
def opt(long, short, desc, **kwargs):
    return option(long, short, help=desc, action="store_true", **kwargs)


# arg decorator
arg = option


# combines option decorators
def option_group(*options):
    def _decorator(func):
        for option in options:
            func = option(func)
        return func

    return _decorator


class MetaCommander(type):
    def __new__(cls, classname, bases, classdict):
        classdict = dict(classdict)
        subcmds = {}
        for name, func in list(classdict.items()):
            if name.startswith("do_"):
                name = name[3:]
                subcmd = {"name": name, "func": func, "options": []}
                if hasattr(func, "options"):
                    subcmd["options"] = func.options
                subcmds[name] = subcmd
        classdict["_argparse_subcmds"] = subcmds
        return type.__new__(cls, classname, bases, classdict)


class Application(ShellCmd, metaclass=MetaCommander):
    """cyllama build manager"""

    version: str = "0.0.4"
    epilog: str = ""
    default_args: list[str] = ["--help"]
    project: Project
    parser: argparse.ArgumentParser
    options: argparse.Namespace
    _argparse_subcmds: dict  # Added by metaclass

    def __init__(self) -> None:
        self.project = Project()
        self.log = logging.getLogger(self.__class__.__name__)

    def parse_args(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            # prog = self.name,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=self.__doc__,
            epilog=self.epilog,
        )
        return parser

    def cmdline(self) -> None:
        self.parser = self.parse_args()

        self.parser.add_argument(
            "-v", "--version", action="version", version="%(prog)s " + self.version
        )

        subparsers = self.parser.add_subparsers(
            title="subcommands",
            description="valid subcommands",
            help="additional help",
            metavar="",
        )

        for name in sorted(self._argparse_subcmds.keys()):
            subcmd = self._argparse_subcmds[name]
            subparser = subparsers.add_parser(
                subcmd["name"], help=subcmd["func"].__doc__
            )
            for args, kwds in subcmd["options"]:
                subparser.add_argument(*args, **kwds)
            subparser.set_defaults(func=subcmd["func"])

        if len(sys.argv) <= 1:
            options = self.parser.parse_args(self.default_args)
        else:
            options = self.parser.parse_args()

        self.options = options
        options.func(self, options)

    # ------------------------------------------------------------------------
    # setup

    def do_setup(self, args: argparse.Namespace) -> None:
        """setup prerequisites"""
        # for Builder in [LlamaCppBuilder, WhisperCppBuilder, StableDiffusionCppBuilder]:
        for Builder in [LlamaCppBuilder]:
            builder = Builder()
            builder.setup()

    # ------------------------------------------------------------------------
    # build

    @opt("--metal", "-m", "enable Metal backend (macOS)")
    @opt("--cuda", "-c", "enable CUDA backend (NVIDIA GPUs)")
    @opt("--vulkan", "-V", "enable Vulkan backend (cross-platform)")
    @opt("--sycl", "-y", "enable SYCL backend (Intel GPUs)")
    @opt("--hip", "-H", "enable HIP/ROCm backend (AMD GPUs)")
    @opt("--opencl", "-o", "enable OpenCL backend")
    @opt("--cpu-only", "-C", "disable all GPU backends (CPU only)")
    @opt("--sd-metal", "-M", "enable Metal for stable-diffusion.cpp (default on macOS)")
    @opt("-w", "--whisper-cpp", "build whisper-cpp")
    @opt("-d", "--stable-diffusion", "build stable-diffusion")
    @opt("-l", "--llama-cpp", "build llama-cpp")
    @opt("-v", "--sqlite-vector", "build sqlite-vector")
    @opt("-s", "--shared", "build shared libraries")
    @opt("-a", "--all", "build all")
    @opt("-D", "--deps-only", "build dependencies only, skip editable install")
    @option(
        "--llama-version",
        default=LLAMACPP_VERSION,
        help=f"llama.cpp version (default: {LLAMACPP_VERSION})",
    )
    @option(
        "--whisper-version",
        default=WHISPERCPP_VERSION,
        help=f"whisper.cpp version (default: {WHISPERCPP_VERSION})",
    )
    @option(
        "--sd-version",
        default=SDCPP_VERSION,
        help=f"stable-diffusion.cpp version (default: {SDCPP_VERSION})",
    )
    @option(
        "--vector-version",
        default=SQLITEVECTOR_VERSION,
        help=f"sqlite-vector version (default: {SQLITEVECTOR_VERSION})",
    )
    def do_build(self, args: argparse.Namespace) -> None:
        """build packages"""
        # Set backend environment variables based on command-line args
        if args.cpu_only:
            os.environ["GGML_METAL"] = "0"
            os.environ["GGML_CUDA"] = "0"
            os.environ["GGML_VULKAN"] = "0"
            os.environ["GGML_SYCL"] = "0"
            os.environ["GGML_HIP"] = "0"
            os.environ["GGML_OPENCL"] = "0"
            os.environ["SD_METAL"] = "0"
        else:
            if args.metal:
                os.environ["GGML_METAL"] = "1"
            if args.cuda:
                os.environ["GGML_CUDA"] = "1"
            if args.vulkan:
                os.environ["GGML_VULKAN"] = "1"
            if args.sycl:
                os.environ["GGML_SYCL"] = "1"
            if args.hip:
                os.environ["GGML_HIP"] = "1"
            if args.opencl:
                os.environ["GGML_OPENCL"] = "1"
            if args.sd_metal:
                os.environ["SD_METAL"] = "1"

        # Map builder classes to their version arguments
        builder_versions = {
            LlamaCppBuilder: args.llama_version,
            WhisperCppBuilder: args.whisper_version,
            StableDiffusionCppBuilder: args.sd_version,
            SqliteVectorBuilder: args.vector_version,
        }

        _builders = []

        if args.all:
            _builders = [
                LlamaCppBuilder,
                WhisperCppBuilder,
                StableDiffusionCppBuilder,
                SqliteVectorBuilder,
            ]
        else:
            if args.llama_cpp:
                _builders.append(LlamaCppBuilder)
            if args.whisper_cpp:
                _builders.append(WhisperCppBuilder)
            if args.stable_diffusion:
                _builders.append(StableDiffusionCppBuilder)
            if args.sqlite_vector:
                _builders.append(SqliteVectorBuilder)

        for BuilderClass in _builders:
            version = builder_versions.get(BuilderClass)
            builder = BuilderClass(version=version)
            builder.build()

        # Build using scikit-build-core (editable install)
        if not args.deps_only:
            _cmd = "uv pip install -e ."
            self.cmd(_cmd)

    # ------------------------------------------------------------------------
    # wheel

    @opt("--release", "-r", "build and release all wheels")
    @opt("--build", "-b", "build single wheel based on STATIC env var")
    @opt("--dynamic", "-d", "build dynamic variant")
    @opt("--static", "-s", "build static variant")
    @opt("--universal", "-u", "build universal wheel")
    @opt("--test", "-t", "test built wheels")
    def do_wheel(self, args: argparse.Namespace) -> None:
        """build wheels"""

        if args.release:
            b = WheelBuilder(universal=args.universal)
            b.release()

        elif args.build:
            b = WheelBuilder(universal=args.universal)
            b.build()

        elif args.dynamic:
            b = WheelBuilder(universal=args.universal)
            b.build_dynamic_wheel()
            b.check()
            b.clean()

        elif args.static:
            b = WheelBuilder(universal=args.universal)
            b.build_static_wheel()
            b.check()
            b.clean()

        if args.test:
            b = WheelBuilder()
            b.test_wheels()

    # ------------------------------------------------------------------------
    # test

    @opt("--pytest", "-p", "run pytest")
    def do_test(self, args: argparse.Namespace) -> None:
        """test modules"""
        if args.pytest:
            self.cmd("pytest -vv tests")
        else:
            for t in self.project.tests.glob("test_*.py"):
                self.cmd(f'"{PYTHON}" {t}')

    # ------------------------------------------------------------------------
    # clean

    @opt("--reset", "-r", "reset project (removes build/ and thirdparty libs)")
    @opt("--verbose", "-v", "verbose cleaning ops")
    def do_clean(self, args: argparse.Namespace) -> None:
        """clean build artifacts"""
        cwd = self.project.cwd
        src = cwd / "src" / "cyllama"
        verbose = args.verbose

        # Directories to remove
        dir_targets = ["dist", ".coverage"]
        dir_pats = ["build/lib.*", "build/temp.*", "build/cp*"]

        # Glob patterns
        glob_pats = [".*_cache", "*.egg-info", "__pycache__", ".DS_Store"]

        # Generated Cython .cpp files
        cython_cpp_files = [
            src / "llama" / "llama_cpp.cpp",
            src / "llama" / "server" / "embedded.cpp",
            src / "whisper" / "whisper_cpp.cpp",
            src / "sd" / "stable_diffusion.cpp",
        ]

        # Clean directories
        for t in dir_targets:
            self.remove(cwd / t, silent=not verbose)

        # Clean directory patterns
        for pat in dir_pats:
            for m in cwd.glob(pat):
                self.remove(m, silent=not verbose)

        # Clean glob patterns recursively
        for p in glob_pats:
            for m in cwd.glob(p):
                self.remove(m, silent=not verbose)
            for m in cwd.glob("**/" + p):
                self.remove(m, silent=not verbose)

        # Clean .so files
        for so in src.glob("*.so"):
            self.remove(so, silent=not verbose)
        for so in src.glob("**/*.so"):
            self.remove(so, silent=not verbose)

        # Clean generated Cython .cpp files
        for cpp in cython_cpp_files:
            if cpp.exists():
                self.remove(cpp, silent=not verbose)

        # Reset: also remove build/ and thirdparty libs
        if args.reset:
            self.remove(cwd / "build", silent=not verbose)

            thirdparty = cwd / "thirdparty"
            for dep in ["llama.cpp", "whisper.cpp", "stable-diffusion.cpp"]:
                dep_dir = thirdparty / dep
                for subdir in ["bin", "lib", "include"]:
                    self.remove(dep_dir / subdir, silent=not verbose)

        self.log.info("Clean complete")

    # ------------------------------------------------------------------------
    # info

    @opt("--snapshot", "-s", "commit and push with dependency versions")
    def do_info(self, args: argparse.Namespace) -> None:
        """show version info for dependencies"""
        build_dir = self.project.cwd / "build"
        deps = [
            ("llama.cpp", build_dir / "llama.cpp"),
            ("whisper.cpp", build_dir / "whisper.cpp"),
            ("sd.cpp", build_dir / "stable-diffusion.cpp"),
            ("sqlite-vector", build_dir / "sqlite-vector"),
        ]

        versions = []
        for name, src_dir in deps:
            if not src_dir.exists():
                if not args.snapshot:
                    self.log.info(f"{name}: not downloaded")
                continue

            # Get git info
            try:
                short = subprocess.run(
                    ["git", "rev-parse", "--short", "HEAD"],
                    cwd=src_dir, capture_output=True, text=True, check=True
                ).stdout.strip()

                tag_result = subprocess.run(
                    ["git", "tag", "--points-at", "HEAD"],
                    cwd=src_dir, capture_output=True, text=True, check=True
                )
                tag = tag_result.stdout.strip().split('\n')[0] if tag_result.stdout.strip() else ""

                if tag:
                    versions.append(f"{name}:{tag}")
                    if not args.snapshot:
                        self.log.info(f"{name}: tag={tag} commit={short}")
                else:
                    versions.append(f"{name}:{short}")
                    if not args.snapshot:
                        self.log.info(f"{name}: commit={short}")
            except (subprocess.CalledProcessError, FileNotFoundError):
                self.log.warning(f"{name}: unable to get git info")

        # Handle --snapshot: commit and push with version info
        if args.snapshot:
            if not versions:
                self.log.error("No dependencies found, cannot create snapshot")
                return

            version_str = " ".join(versions)
            commit_msg = f"synced to {version_str}"

            self.log.info(f"Creating snapshot: {commit_msg}")

            try:
                # git add --all
                subprocess.run(
                    ["git", "add", "--all", "."],
                    cwd=self.project.cwd, check=True
                )

                # git commit
                subprocess.run(
                    ["git", "commit", "-m", commit_msg],
                    cwd=self.project.cwd, check=True
                )

                # git push
                subprocess.run(
                    ["git", "push"],
                    cwd=self.project.cwd, check=True
                )

                self.log.info("Snapshot complete")
            except subprocess.CalledProcessError as e:
                self.log.error(f"Snapshot failed: {e}")

    # ------------------------------------------------------------------------
    # bump

    @opt("--major", "-M", "increment major version (X.0.0)")
    @opt("--minor", "-m", "increment minor version (0.X.0)")
    @opt("--dry-run", "-n", "show what would be done without making changes")
    def do_bump(self, args: argparse.Namespace) -> None:
        """bump version and create git tag"""
        import re

        # Files containing version
        pyproject_path = self.project.cwd / "pyproject.toml"
        init_path = self.project.cwd / "src" / "cyllama" / "__init__.py"

        # Read current version from pyproject.toml
        pyproject_content = pyproject_path.read_text()
        version_match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject_content, re.MULTILINE)
        if not version_match:
            self.log.error("Could not find version in pyproject.toml")
            return

        current_version = version_match.group(1)
        self.log.info(f"Current version: {current_version}")

        # Parse semantic version
        parts = current_version.split(".")
        if len(parts) != 3:
            self.log.error(f"Invalid semantic version format: {current_version}")
            return

        try:
            major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            self.log.error(f"Invalid version numbers: {current_version}")
            return

        # Calculate new version
        if args.major:
            major += 1
            minor = 0
            patch = 0
        elif args.minor:
            minor += 1
            patch = 0
        else:
            # Default: patch increment
            patch += 1

        new_version = f"{major}.{minor}.{patch}"
        self.log.info(f"New version: {new_version}")

        if args.dry_run:
            self.log.info("Dry run - no changes made")
            self.log.info(f"Would update: {pyproject_path}")
            self.log.info(f"Would update: {init_path}")
            self.log.info(f"Would create git tag: {new_version}")
            return

        # Update pyproject.toml
        new_pyproject = re.sub(
            r'^(version\s*=\s*)"[^"]+"',
            f'\\1"{new_version}"',
            pyproject_content,
            flags=re.MULTILINE
        )
        pyproject_path.write_text(new_pyproject)
        self.log.info(f"Updated {pyproject_path}")

        # Update __init__.py
        init_content = init_path.read_text()
        new_init = re.sub(
            r'^(__version__\s*=\s*)"[^"]+"',
            f'\\1"{new_version}"',
            init_content,
            flags=re.MULTILINE
        )
        init_path.write_text(new_init)
        self.log.info(f"Updated {init_path}")

        # Git operations
        try:
            # Stage version files
            subprocess.run(
                ["git", "add", str(pyproject_path), str(init_path)],
                cwd=self.project.cwd, check=True
            )

            # Commit
            subprocess.run(
                ["git", "commit", "-m", f"bump version to {new_version}"],
                cwd=self.project.cwd, check=True
            )

            # Create tag
            subprocess.run(
                ["git", "tag", new_version],
                cwd=self.project.cwd, check=True
            )
            self.log.info(f"Created git tag: {new_version}")

            # Push commit and tag
            subprocess.run(
                ["git", "push"],
                cwd=self.project.cwd, check=True
            )
            subprocess.run(
                ["git", "push", "origin", "tag", new_version],
                cwd=self.project.cwd, check=True
            )
            self.log.info(f"Pushed tag {new_version} to origin")

        except subprocess.CalledProcessError as e:
            self.log.error(f"Git operation failed: {e}")
            return

        self.log.info(f"Version bump complete: {current_version} -> {new_version}")

    # ------------------------------------------------------------------------
    # download

    @opt("--llama", "-l", "download default llama model")
    @opt("--whisper", "-w", "download whisper model")
    @option("--whisper-model", "-W", default="base.en", help="whisper model name (default: base.en)")
    @option("--models-dir", "-d", default="models", help="models directory (default: models)")
    def do_download(self, args: argparse.Namespace) -> None:
        """download models"""
        models_dir = Path(args.models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)

        if args.llama:
            # Download default llama model
            model_name = "Llama-3.2-1B-Instruct-Q8_0.gguf"
            model_path = models_dir / model_name
            if model_path.exists():
                self.log.info(f"Model already exists: {model_path}")
            else:
                url = f"https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/{model_name}"
                self.log.info(f"Downloading {model_name}...")
                urlretrieve(url, model_path)
                self.log.info(f"Downloaded to {model_path}")

        if args.whisper:
            # Download whisper model
            model_name = args.whisper_model
            valid_models = [
                "tiny", "tiny.en", "tiny-q5_1", "tiny.en-q5_1", "tiny-q8_0",
                "base", "base.en", "base-q5_1", "base.en-q5_1", "base-q8_0",
                "small", "small.en", "small.en-tdrz", "small-q5_1", "small.en-q5_1", "small-q8_0",
                "medium", "medium.en", "medium-q5_0", "medium.en-q5_0", "medium-q8_0",
                "large-v1", "large-v2", "large-v2-q5_0", "large-v2-q8_0",
                "large-v3", "large-v3-q5_0", "large-v3-turbo", "large-v3-turbo-q5_0", "large-v3-turbo-q8_0"
            ]
            if model_name not in valid_models:
                self.log.error(f"Invalid whisper model: {model_name}")
                self.log.info(f"Available models: {', '.join(valid_models)}")
                return

            model_file = f"ggml-{model_name}.bin"
            model_path = models_dir / model_file
            if model_path.exists():
                self.log.info(f"Model already exists: {model_path}")
            else:
                if "tdrz" in model_name:
                    src = "https://huggingface.co/akashmjn/tinydiarize-whisper.cpp"
                else:
                    src = "https://huggingface.co/ggerganov/whisper.cpp"
                url = f"{src}/resolve/main/ggml-{model_name}.bin"
                self.log.info(f"Downloading ggml-{model_name}.bin...")
                urlretrieve(url, model_path)
                self.log.info(f"Downloaded to {model_path}")

        if not args.llama and not args.whisper:
            self.log.info("Specify --llama or --whisper to download models")
            self.log.info("  --llama: Download Llama-3.2-1B-Instruct-Q8_0.gguf")
            self.log.info("  --whisper: Download whisper model (use --whisper-model to specify)")

    # ------------------------------------------------------------------------
    # bins

    @opt("--clean", "-c", "clean before building")
    def do_bins(self, args: argparse.Namespace) -> None:
        """build llama.cpp CLI binaries"""
        build_dir = self.project.cwd / "build" / "llama.cpp"
        prefix = self.project.cwd / "thirdparty" / "llama.cpp"
        bin_dir = prefix / "bin"

        if not build_dir.exists():
            self.log.error(f"llama.cpp source not found at {build_dir}")
            self.log.info("Run 'python manage.py build -l' first to download llama.cpp")
            return

        bins_build_dir = build_dir / "build-bins"

        if args.clean:
            self.log.info("Cleaning previous binary build...")
            self.remove(bins_build_dir)
            self.remove(bin_dir)

        bin_dir.mkdir(parents=True, exist_ok=True)

        # Get backend options
        builder = LlamaCppBuilder()
        backend_options = builder.get_backend_cmake_options()

        # Configure cmake for binaries
        cmake_options = {
            "BUILD_SHARED_LIBS": False,
            "CMAKE_POSITION_INDEPENDENT_CODE": True,
            "LLAMA_BUILD_EXAMPLES": True,
            "LLAMA_BUILD_SERVER": True,
            "LLAMA_BUILD_TESTS": False,
            "LLAMA_CURL": False,
            **backend_options,
        }

        self.log.info(f"Configuring llama.cpp binaries...")
        builder.cmake_config(
            src_dir=build_dir,
            build_dir=bins_build_dir,
            **cmake_options,
        )

        self.log.info("Building binaries...")
        self.cmd(f"cmake --build {bins_build_dir} --config Release -j", cwd=build_dir)

        # Copy binaries
        self.log.info(f"Installing binaries to {bin_dir}...")
        bin_src = bins_build_dir / "bin"
        if bin_src.exists():
            for binary in bin_src.glob("llama-*"):
                if binary.is_file():
                    shutil.copy2(binary, bin_dir)

        # Count installed binaries
        bin_count = len(list(bin_dir.glob("llama-*")))
        self.log.info(f"Installed {bin_count} binaries to {bin_dir}")
        self.log.info(f"Add to PATH: export PATH=\"{bin_dir}:$PATH\"")

    # ------------------------------------------------------------------------
    # profile

    @option("-m", "--model", default="models/Llama-3.2-1B-Instruct-Q8_0.gguf", help="model path")
    @opt("--tokenization", "-t", "profile tokenization")
    @opt("--inference", "-i", "profile inference")
    @opt("--logits", "-l", "profile logits retrieval")
    @opt("--batch", "-b", "profile batch operations")
    @opt("--properties", "-p", "profile property access")
    @opt("--all", "-a", "profile all operations")
    @option("--iterations", "-n", type=int, default=100, help="iterations per test")
    @option("--output", "-o", default=None, help="output directory for profile data")
    def do_profile(self, args: argparse.Namespace) -> None:
        """profile cyllama operations using cProfile"""
        import cProfile
        import pstats
        import io
        import time

        model_path = Path(args.model)
        if not model_path.exists():
            self.log.error(f"Model not found: {model_path}")
            return

        # Import cyllama
        try:
            sys.path.insert(0, str(self.project.cwd / "src"))
            import cyllama
            from cyllama import llama_batch_get_one
        except ImportError as e:
            self.log.error(f"Failed to import cyllama: {e}")
            return

        profiles = {}
        iterations = args.iterations

        # Determine what to profile
        profile_all = args.all or not any([
            args.tokenization, args.inference, args.logits, args.batch, args.properties
        ])

        # Load model once for all tests
        self.log.info(f"Loading model: {model_path}")
        model_params = cyllama.LlamaModelParams()
        model = cyllama.LlamaModel(str(model_path), model_params)
        vocab = model.get_vocab()

        # Profile tokenization
        if profile_all or args.tokenization:
            print("\n=== Profiling Tokenization ===")
            test_texts = [
                "Hello world",
                "This is a longer sentence to tokenize.",
                "Machine learning and AI " * 5,
            ]

            def tokenize_benchmark():
                total = 0
                for text in test_texts:
                    for _ in range(iterations):
                        tokens = vocab.tokenize(text, add_special=True, parse_special=False)
                        total += len(tokens)
                return total

            pr = cProfile.Profile()
            pr.enable()
            t0 = time.perf_counter()
            total_tokens = tokenize_benchmark()
            elapsed = time.perf_counter() - t0
            pr.disable()

            print(f"Tokenized {total_tokens} tokens in {elapsed:.3f}s ({total_tokens/elapsed:.0f} tokens/s)")
            self._print_profile_stats(pr, 10)
            profiles['tokenization'] = pr

        # Profile inference
        if profile_all or args.inference:
            print("\n=== Profiling Inference ===")

            ctx_params = cyllama.LlamaContextParams()
            ctx_params.n_ctx = 256
            ctx_params.n_batch = 512
            ctx = cyllama.LlamaContext(model, ctx_params)

            sampler_params = cyllama.LlamaSamplerChainParams()
            sampler = cyllama.LlamaSampler(sampler_params)
            sampler.add_greedy()

            prompt_tokens = vocab.tokenize("The future of AI is", add_special=True, parse_special=False)

            def inference_benchmark():
                generated = 0
                for _ in range(min(iterations // 10, 10)):  # Fewer iterations, inference is slow
                    ctx.kv_cache_clear()
                    batch = llama_batch_get_one(prompt_tokens)
                    ctx.decode(batch)
                    for _ in range(20):  # Generate 20 tokens
                        token = sampler.sample(ctx, -1)
                        if model.token_is_eog(token):
                            break
                        sampler.accept(token)
                        batch = llama_batch_get_one([token])
                        ctx.decode(batch)
                        generated += 1
                return generated

            pr = cProfile.Profile()
            pr.enable()
            t0 = time.perf_counter()
            total_generated = inference_benchmark()
            elapsed = time.perf_counter() - t0
            pr.disable()

            print(f"Generated {total_generated} tokens in {elapsed:.3f}s ({total_generated/elapsed:.1f} tokens/s)")
            self._print_profile_stats(pr, 10)
            profiles['inference'] = pr

        # Profile logits retrieval
        if profile_all or args.logits:
            print("\n=== Profiling Logits Retrieval ===")

            ctx_params = cyllama.LlamaContextParams()
            ctx_params.n_ctx = 128
            ctx = cyllama.LlamaContext(model, ctx_params)

            tokens = vocab.tokenize("Test", add_special=True, parse_special=False)
            batch = llama_batch_get_one(tokens)
            ctx.decode(batch)

            def logits_benchmark():
                total = 0
                for _ in range(iterations):
                    logits = ctx.get_logits()
                    total += len(logits)
                return total

            pr = cProfile.Profile()
            pr.enable()
            t0 = time.perf_counter()
            total_logits = logits_benchmark()
            elapsed = time.perf_counter() - t0
            pr.disable()

            print(f"Retrieved {total_logits:,} logit values in {elapsed:.3f}s ({total_logits/elapsed:,.0f} values/s)")
            self._print_profile_stats(pr, 10)
            profiles['logits'] = pr

        # Profile batch operations
        if profile_all or args.batch:
            print("\n=== Profiling Batch Operations ===")

            test_tokens = list(range(100))

            def batch_benchmark():
                total = 0
                for _ in range(iterations * 10):
                    batch = llama_batch_get_one(test_tokens)
                    total += 1
                return total

            pr = cProfile.Profile()
            pr.enable()
            t0 = time.perf_counter()
            total_batches = batch_benchmark()
            elapsed = time.perf_counter() - t0
            pr.disable()

            print(f"Created {total_batches} batches in {elapsed:.3f}s ({total_batches/elapsed:.0f} batches/s)")
            self._print_profile_stats(pr, 10)
            profiles['batch'] = pr

        # Profile property access
        if profile_all or args.properties:
            print("\n=== Profiling Property Access ===")

            def properties_benchmark():
                total = 0
                for _ in range(iterations * 10):
                    total += model.n_embd
                    total += model.n_layer
                    total += model.n_vocab
                return total

            pr = cProfile.Profile()
            pr.enable()
            t0 = time.perf_counter()
            result = properties_benchmark()
            elapsed = time.perf_counter() - t0
            pr.disable()

            accesses = iterations * 10 * 3
            print(f"{accesses} property accesses in {elapsed:.3f}s ({accesses/elapsed:.0f} accesses/s)")
            self._print_profile_stats(pr, 10)
            profiles['properties'] = pr

        # Save profile data if output directory specified
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            for name, pr in profiles.items():
                pr.dump_stats(output_dir / f"{name}_profile.prof")
                self.log.info(f"Saved {name} profile to {output_dir / f'{name}_profile.prof'}")

        print("\n" + "=" * 50)
        print("Profiling Complete!")
        print("\nKey metrics:")
        print("- cumtime: Total time spent in function and its callees")
        print("- tottime: Time spent in function only (excluding callees)")
        print("- ncalls: Number of times the function was called")

    def _print_profile_stats(self, pr, n: int = 10):
        """Print top N functions from profile."""
        import pstats
        import io
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(n)
        for line in s.getvalue().split('\n')[:n + 10]:
            print(line)

    # ------------------------------------------------------------------------
    # bench

    @option("-m", "--model", default="models/Llama-3.2-1B-Instruct-Q8_0.gguf", help="model path")
    @option("-p", "--prompt", default="Explain the theory of relativity in simple terms.", help="prompt")
    @option("-n", "--n-tokens", type=int, default=100, help="tokens to generate")
    @option("-r", "--runs", type=int, default=3, help="number of runs")
    @opt("--no-warmup", "-W", "skip warmup run")
    def do_bench(self, args: argparse.Namespace) -> None:
        """run performance benchmark"""
        import statistics
        import time

        model_path = Path(args.model)
        if not model_path.exists():
            self.log.error(f"Model not found: {model_path}")
            self.log.info("Run 'python manage.py download --llama' to download default model")
            return

        # Import cyllama
        try:
            sys.path.insert(0, str(self.project.cwd / "src"))
            from cyllama.llama.llama_cpp import (
                LlamaModel,
                LlamaContext,
                LlamaModelParams,
                LlamaContextParams,
                LlamaSampler,
                LlamaSamplerChainParams,
                llama_batch_get_one,
            )
        except ImportError as e:
            self.log.error(f"Failed to import cyllama: {e}")
            self.log.info("Run 'python manage.py build -l' first")
            return

        self.log.info(f"Loading model: {model_path}")
        model_params = LlamaModelParams()
        model = LlamaModel(path_model=str(model_path), params=model_params)

        ctx_params = LlamaContextParams()
        ctx_params.n_ctx = 2048
        ctx_params.n_batch = 512
        ctx = LlamaContext(model=model, params=ctx_params)

        sampler_params = LlamaSamplerChainParams()
        sampler = LlamaSampler(sampler_params)
        sampler.add_greedy()

        prompt_tokens = model.tokenize(args.prompt.encode(), add_bos=True, special=True)
        n_prompt = len(prompt_tokens)

        results = []

        # Warmup
        if not args.no_warmup:
            self.log.info("Warmup run...")
            ctx.kv_cache_clear()
            batch = llama_batch_get_one(prompt_tokens)
            ctx.decode(batch)
            for _ in range(10):
                token = sampler.sample(ctx, -1)
                sampler.accept(token)
                batch = llama_batch_get_one([token])
                ctx.decode(batch)

        self.log.info(f"Running {args.runs} benchmark iterations...")

        for run in range(args.runs):
            ctx.kv_cache_clear()

            # Prefill
            t0 = time.perf_counter()
            batch = llama_batch_get_one(prompt_tokens)
            ctx.decode(batch)
            prefill_time = (time.perf_counter() - t0) * 1000

            # Decode
            t0 = time.perf_counter()
            generated = 0
            for _ in range(args.n_tokens):
                token = sampler.sample(ctx, -1)
                if model.token_is_eog(token):
                    break
                sampler.accept(token)
                batch = llama_batch_get_one([token])
                ctx.decode(batch)
                generated += 1
            decode_time = (time.perf_counter() - t0) * 1000

            prefill_speed = n_prompt / (prefill_time / 1000)
            decode_speed = generated / (decode_time / 1000) if decode_time > 0 else 0

            results.append({
                "prefill_ms": prefill_time,
                "decode_ms": decode_time,
                "prefill_tps": prefill_speed,
                "decode_tps": decode_speed,
                "generated": generated,
            })

            self.log.info(f"  Run {run + 1}: prefill={prefill_speed:.1f} t/s, decode={decode_speed:.1f} t/s")

        # Summary
        avg_prefill = statistics.mean(r["prefill_tps"] for r in results)
        avg_decode = statistics.mean(r["decode_tps"] for r in results)
        std_prefill = statistics.stdev(r["prefill_tps"] for r in results) if len(results) > 1 else 0
        std_decode = statistics.stdev(r["decode_tps"] for r in results) if len(results) > 1 else 0

        print()
        print("=" * 50)
        print(f"Benchmark Results ({args.runs} runs)")
        print("=" * 50)
        print(f"Model: {model_path.name}")
        print(f"Prompt tokens: {n_prompt}")
        print(f"Generated tokens: {results[0]['generated']}")
        print()
        print(f"Prefill: {avg_prefill:.1f} +/- {std_prefill:.1f} tokens/sec")
        print(f"Decode:  {avg_decode:.1f} +/- {std_decode:.1f} tokens/sec")
        print("=" * 50)


if __name__ == "__main__":
    Application().cmdline()
