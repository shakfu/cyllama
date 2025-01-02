#!/usr/bin/env python3

"""manage.py: cross-platform cyllama build manager.

It only uses python stdlib modules to do the following:

- General Shell ops
- Dependency download, build, install
- Module compilation
- Wheel building
- Alternative frontend to Makefile
- Downloads/build a local version python for testing

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

    clean        clean detritus
    build        build application
    setup        setup prerequisites
    test         test modules
    wheel        build wheels

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
from fnmatch import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union, Callable
from urllib.request import urlretrieve

PYTHON = sys.executable
PLATFORM = platform.system()
ARCH = platform.machine()
PY_VER_MINOR = sys.version_info.minor

__version__ = "0.0.1"

# ----------------------------------------------------------------------------
# type aliases

Pathlike = Union[str, Path]
MatchFn = Callable[[Path], bool]
ActionFn = Callable[[Path], None]

# ----------------------------------------------------------------------------
# env helpers


def getenv(key: str, default: bool = False) -> bool:
    """convert '0','1' env values to bool {True, False}"""
    return bool(int(os.getenv(key, default)))


def setenv(key: str, default: str):
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
if PLATFORM == "Darwin":
    MACOSX_DEPLOYMENT_TARGET = setenv("MACOSX_DEPLOYMENT_TARGET", "12.6")
DEFAULT_PY_VERSION = "3.12.7"
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

    def format(self, record):
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

    def cmd(self, shellcmd: str, cwd: Pathlike = "."):
        """Run shell command within working directory"""
        self.log.info(shellcmd)
        try:
            subprocess.check_call(shellcmd, shell=True, cwd=str(cwd))
        except subprocess.CalledProcessError:
            self.log.critical("", exc_info=True)
            sys.exit(1)

    def download(self, url: str, tofolder: Optional[Pathlike] = None) -> Pathlike:
        """Download a file from a url to an optional folder"""
        _path = Path(os.path.basename(url))
        if tofolder:
            _path = Path(tofolder).joinpath(_path)
            if _path.exists():
                return _path
        filename, _ = urlretrieve(url, filename=_path)
        return Path(filename)

    def extract(self, archive: Pathlike, tofolder: Pathlike = "."):
        """extract a tar archive"""
        if tarfile.is_tarfile(archive):
            with tarfile.open(archive) as f:
                f.extractall(tofolder)
        # elif zipfile.is_zipfile(archive):
        #     with zipfile.ZipFile(archive) as f:
        #         f.extractall(tofolder)
        else:
            raise TypeError("cannot extract from this file.")

    def fail(self, msg, *args):
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
    ):
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

    def chdir(self, path: Pathlike):
        """Change current workding directory to path"""
        self.log.info("changing working dir to: %s", path)
        os.chdir(path)

    def chmod(self, path: Pathlike, perm=0o777):
        """Change permission of file"""
        self.log.info("change permission of %s to %s", path, perm)
        os.chmod(path, perm)

    def get(self, shellcmd, cwd: Pathlike = ".", shell: bool = False) -> str:
        """get output of shellcmd"""
        if not shell:
            shellcmd = shellcmd.split()
        return subprocess.check_output(
            shellcmd, encoding="utf8", shell=shell, cwd=str(cwd)
        ).strip()

    def makedirs(self, path: Pathlike, mode: int = 511, exist_ok: bool = True):
        """Recursive directory creation function"""
        self.log.info("making directory: %s", path)
        os.makedirs(path, mode, exist_ok)

    def move(self, src: Pathlike, dst: Pathlike):
        """Move from src path to dst path."""
        self.log.info("move path %s to %s", src, dst)
        shutil.move(src, dst)

    def copy(self, src: Pathlike, dst: Pathlike):
        """copy file or folders -- tries to be behave like `cp -rf`"""
        self.log.info("copy %s to %s", src, dst)
        src, dst = Path(src), Path(dst)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    def remove(self, path: Pathlike, silent: bool = False):
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
    ):
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
    ):
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

    def glob_remove(self, root: Pathlike, patterns: list[str], skip_dirs: list[str]):
        """applies recursive glob remove using a list of patterns"""

        def _match(entry: Path) -> bool:
            # return any(fnmatch(entry, p) for p in patterns)
            return any(fnmatch(entry.name, p) for p in patterns)

        def remove(entry: Path):
            self.remove(entry)

        self.walk(root, match_func=_match, action_func=remove, skip_patterns=skip_dirs)

    def pip_install(
        self,
        *pkgs,
        reqs: Optional[str] = None,
        upgrade: bool = False,
        pip: Optional[str] = None,
    ):
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

    def apt_install(self, *pkgs, update: bool = False):
        """install debian packages using apt"""
        _cmds = []
        _cmds.append("sudo apt install")
        if update:
            _cmds.append("--upgrade")
        _cmds.extend(pkgs)
        self.cmd(" ".join(_cmds))

    def brew_install(self, *pkgs, update: bool = False):
        """install using homebrew"""
        _pkgs = " ".join(pkgs)
        if update:
            self.cmd("brew update")
        self.cmd(f"brew install {_pkgs}")

    def cmake_config(self, src_dir: Pathlike, build_dir: Pathlike, *scripts, **options):
        """activate cmake configuration / generation stage"""
        _cmds = [f"cmake -S {src_dir} -B {build_dir}"]
        if scripts:
            _cmds.append(" ".join(f"-C {path}" for path in scripts))
        if options:
            _cmds.append(" ".join(f"-D{k}={v}" for k, v in options.items()))
        self.cmd(" ".join(_cmds))

    def cmake_build(self, build_dir: Pathlike, release: bool = False):
        """activate cmake build stage"""
        _cmd = f"cmake --build {build_dir}"
        if release:
            _cmd += " --config Release"
        self.cmd(_cmd)

    def cmake_install(self, build_dir: Pathlike, prefix: Optional[str] = None):
        """activate cmake install stage"""
        _cmds = ["cmake --install", str(build_dir)]
        if prefix:
            _cmds.append(f"--prefix {prefix}")
        self.cmd(" ".join(_cmds))


# ----------------------------------------------------------------------------
# main classes


class Project(ShellCmd):
    """Utility class to hold project directory structure"""

    def __init__(self):
        self.cwd = Path.cwd()
        self.build = self.cwd / "build"
        self.src = self.build / "src"
        self.install = self.cwd / "thirdparty"
        self.bin = self.cwd / "bin"
        self.dist = self.cwd / "dist"
        self.scripts = self.cwd / "scripts"
        self.tests = self.cwd / "tests"

    def setup(self):
        """create main project directories"""
        # self.bin.mkdir(exist_ok=True)
        self.build.mkdir(exist_ok=True)
        self.src.mkdir(exist_ok=True)
        self.install.mkdir(exist_ok=True)

    def reset(self):
        """prepare project for a rebuild"""
        self.remove(self.bin)
        self.remove(self.build)
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
        # return self.project.build / self.name
        return self.project.src / self.name
        # return self.project.src / self.name_version

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

    def libs_static_exist(self) -> bool:
        """check if all built stati libs already exist"""
        return all((self.lib / lib).exists() for lib in self.libs_static)

    def pre_process(self):
        """override by subclass if needed"""

    def setup(self):
        """setup build environment"""

    def configure(self):
        """configure build"""

    def build(self):
        """build target"""

    def install(self):
        """install target"""

    def clean(self):
        """clean build"""

    def post_process(self):
        """override by subclass if needed"""

    def process(self):
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

    def setup(self):
        """setup build environment"""
        self.project.setup()
        archive = self.download(self.download_url, tofolder=self.project.downloads)
        self.log.info("downloaded %s", archive)
        if not self.src_dir.exists():
            self.extract(archive, tofolder=self.project.src)
            assert self.src_dir.exists(), f"could not extract from {archive}"


class LlamaCppBuilder(Builder):
    """build llama.cpp"""

    name: str = "llama.cpp"
    version: str = "b4393"
    repo_url: str = "https://github.com/ggerganov/llama.cpp.git"
    libs_static: list[str] = [
        "libcommon.a",
        "libggml-base.a",
        "libggml-blas.a",
        "libggml-cpu.a",
        "libggml-metal.a",
        "libggml.a",
        "libllama.a",
        "libllava.a",
    ]

    def build(self):
        """llama.cpp main build function"""
        self.log.info("update from llama.cpp main repo")
        self.project.setup()
        self.prefix.mkdir(exist_ok=True)
        self.include.mkdir(exist_ok=True)
        if self.version:
            self.git_clone(
                self.repo_url, branch=self.version, recurse=True, cwd=self.project.src
            )
        else:
            self.git_clone(self.repo_url, recurse=True, cwd=self.project.src)
        self.glob_copy(self.src_dir / "common", self.include, patterns=["*.h", "*.hpp"])
        self.glob_copy(
            self.src_dir / "ggml" / "include", self.include, patterns=["*.h"]
        )
        self.glob_copy(
            self.src_dir / "examples" / "llava", self.include, patterns=["*.h"]
        )
        self.cmake_config(
            src_dir=self.src_dir,
            build_dir=self.build_dir,
            BUILD_SHARED_LIBS=False,
            CMAKE_POSITION_INDEPENDENT_CODE=True,
        )
        self.cmake_build(build_dir=self.build_dir, release=True)
        self.cmake_install(build_dir=self.build_dir, prefix=self.prefix)
        self.copy(self.build_dir / "ggml" / "src" / "libggml-base.a", self.lib)
        self.copy(self.build_dir / "ggml" / "src" / "libggml-cpu.a", self.lib)
        self.copy(
            self.build_dir / "ggml" / "src" / "ggml-blas" / "libggml-blas.a", self.lib
        )
        self.copy(
            self.build_dir / "ggml" / "src" / "ggml-metal" / "libggml-metal.a", self.lib
        )
        self.copy(self.build_dir / "common" / "libcommon.a", self.lib)
        self.copy(
            self.build_dir / "examples" / "llava" / "libllava_static.a",
            self.lib / "libllava.a",
        )
        self.move(self.prefix / "bin", self.project.bin)


class LlamaCppPythonBuilder(Builder):
    """build llama-cpp-python"""

    name: str = "llama-cpp-python"
    version: str = ""
    repo_url: str = "https://github.com/abetlen/llama-cpp-python.git"
    libs_static: list[str] = []

    def build(self):
        """llama-cpp-python main build function"""
        self.log.info("update from llama-cpp-python main repo")
        self.project.setup()
        if self.version:
            self.git_clone(
                self.repo_url, branch=self.version, recurse=True, cwd=self.project.src
            )
        else:
            self.git_clone(self.repo_url, recurse=True, cwd=self.project.src)
        self.copy(self.src_dir / "llama_cpp", self.prefix)


class WhisperCppBuilder(Builder):
    """build whisper.cpp"""

    name: str = "whisper.cpp"
    version: str = ""
    repo_url: str = "https://github.com/ggerganov/whisper.cpp.git"
    libs_static: list[str] = [
        "libcommon.a",
        "libwhisper.a",
        "libggml.a",
    ]

    def build(self):
        """whisper.cpp main build function"""
        self.log.info("update from whisper.cpp main repo")
        self.project.setup()
        self.prefix.mkdir(exist_ok=True)
        self.include.mkdir(exist_ok=True)
        if self.version:
            self.git_clone(
                self.repo_url, branch=self.version, recurse=True, cwd=self.project.src
            )
        else:
            self.git_clone(self.repo_url, recurse=True, cwd=self.project.src)
        self.glob_copy(
            self.src_dir / "examples", self.include, patterns=["*.h", "*.hpp"]
        )
        self.cmake_config(
            src_dir=self.src_dir,
            build_dir=self.build_dir,
            BUILD_SHARED_LIBS=False,
            CMAKE_POSITION_INDEPENDENT_CODE=True,
        )
        self.cmake_build(build_dir=self.build_dir, release=True)
        self.cmake_install(build_dir=self.build_dir, prefix=self.prefix)
        self.copy(self.build_dir / "examples" / "libcommon.a", self.lib)
        self.glob_copy(self.build_dir / "bin", self.bin, patterns=["*"])


class StableDiffusionCppBuilder(Builder):
    """build stable-diffusion.cpp"""

    name: str = "stable-diffusion.cpp"
    version: str = ""
    repo_url: str = "https://github.com/leejet/stable-diffusion.cpp.git"
    libs_static: list[str] = [
        "libstable-diffusion.a",
    ]

    def build(self):
        """stable-diffusion.cpp main build function"""
        self.log.info("update from whisper.cpp main repo")
        self.project.setup()
        self.prefix.mkdir(exist_ok=True)
        self.include.mkdir(exist_ok=True)
        if self.version:
            self.git_clone(
                self.repo_url, branch=self.version, recurse=True, cwd=self.project.src
            )
        else:
            self.git_clone(self.repo_url, recurse=True, cwd=self.project.src)
        self.glob_copy(self.src_dir, self.include, patterns=["*.h", "*.hpp"])
        self.cmake_config(
            src_dir=self.src_dir,
            build_dir=self.build_dir,
            BUILD_SHARED_LIBS=False,
            CMAKE_POSITION_INDEPENDENT_CODE=True,
        )
        self.cmake_build(build_dir=self.build_dir, release=True)
        self.cmake_install(build_dir=self.build_dir, prefix=self.prefix)
        self.copy(self.build_dir / "libstable-diffusion.a", self.lib)


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

    def __init__(self, universal: bool = False):
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
    def is_static(self):
        return self.getenv("STATIC")

    @property
    def is_macos_arm64(self):
        return PLATFORM == "Darwin" and ARCH == "arm64"

    @property
    def is_macos_x86_64(self):
        return PLATFORM == "Darwin" and ARCH == "x86_64"

    @property
    def is_linux_x86_64(self):
        return PLATFORM == "Linux" and ARCH == "x86_64"

    @property
    def is_linux_aarch64(self):
        return PLATFORM == "Linux" and ARCH == "aarch64"

    def clean(self):
        if self.project.build.exists():
            shutil.rmtree(self.project.build, ignore_errors=True)
        if self.project.dist.exists():
            shutil.rmtree(self.project.dist)

    def reset(self):
        self.clean()
        if self.project.wheels.exists():
            shutil.rmtree(self.project.wheels)

    def check(self):
        have_wheels = bool(self.project.wheels.glob("*.whl"))
        if not have_wheels:
            self.fail("no wheels created")

    def makedirs(self):
        if not self.project.wheels.exists():
            self.project.wheels.mkdir()

    def build_wheel(self, static: bool = False, override: bool = True):
        assert PY_VER_MINOR >= 8, "only supporting python >= 3.8"

        _cmd = f'"{PYTHON}" setup.py bdist_wheel'

        if PLATFORM == "Darwin":
            ver = self.get_min_osx_ver()
            if self.universal:
                prefix = (
                    f"ARCHFLAGS='-arch arm64 -arch x86_64' "
                    f"_PYTHON_HOST_PLATFORM='macosx-{ver}-universal2' "
                )
            else:
                prefix = (
                    f"ARCHFLAGS='-arch {ARCH}' "
                    f"_PYTHON_HOST_PLATFORM='macosx-{ver}-{ARCH}' "
                )

            _cmd = prefix + _cmd

        if static:
            os.environ["STATIC"] = "1"
        self.cmd(_cmd)

    def test_wheels(self):
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

    def build_dynamic_wheel(self):
        self.log.info("building dynamic build wheel")
        self.clean()
        self.makedirs()
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

    def build_static_wheel(self):
        self.log.info("building static build wheel")
        self.clean()
        self.makedirs()
        self.build_wheel(static=True)
        for wheel in self.project.dist.glob("*.whl"):
            w = WheelFilename.from_path(wheel)
            w.project = "cyllama-static"
            renamed_wheel = str(w)
            os.rename(wheel, renamed_wheel)
            shutil.move(renamed_wheel, self.project.wheels)

    def build(self):
        if self.is_static:
            self.build_static_wheel()
        else:
            self.build_dynamic_wheel()
        self.check()
        self.clean()

    def release(self):
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
def opt(long, short, desc):
    return option(long, short, help=desc, action="store_true")


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

    version = "0.0.4"
    epilog = ""
    default_args = ["--help"]

    def __init__(self):
        self.project = Project()
        self.log = logging.getLogger(self.__class__.__name__)

    def parse_args(self):
        parser = argparse.ArgumentParser(
            # prog = self.name,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=self.__doc__,
            epilog=self.epilog,
        )
        return parser

    def cmdline(self):
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

    def do_setup(self, args):
        """setup prerequisites"""
        for Builder in [LlamaCppBuilder, LlamaCppPythonBuilder]:
            builder = Builder()
            builder.build()


    # ------------------------------------------------------------------------
    # build

    @opt("--static", "-s", "build static variant")
    def do_build(self, args):
        """build packages"""
        _cmd = f'"{PYTHON}" setup.py build_ext --inplace'
        if args.static:
            os.environ["STATIC"] = "1"
        self.cmd(_cmd)

    # ------------------------------------------------------------------------
    # wheel

    @opt("--release", "-r", "build and release all wheels")
    @opt("--build", "-b", "build single wheel based on STATIC env var")
    @opt("--dynamic", "-d", "build dynamic variant")
    @opt("--static", "-s", "build static variant")
    @opt("--universal", "-u", "build universal wheel")
    @opt("--test", "-t", "test built wheels")
    def do_wheel(self, args):
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
    def do_test(self, args):
        """test modules"""
        if args.pytest:
            self.cmd("pytest -vv tests")
        else:
            for t in self.project.tests.glob("test_*.py"):
                self.cmd(f'"{PYTHON}" {t}')

    # ------------------------------------------------------------------------
    # clean

    @opt("--reset", "-r", "reset project")
    @opt("--verbose", "-v", "verbose cleaning ops")
    def do_clean(self, args):
        """clean detritus"""
        cwd = self.project.cwd
        _targets = ["build", "dist", "venv", "MANIFEST.in", ".task"]
        if args.reset:
            _targets += ["python", "bin", "lib", "share", "wheels"]
        _pats = [".*_cache", "*.egg-info", "__pycache__", ".DS_Store"]
        for t in _targets:
            self.remove(cwd / t, silent=not args.verbose)
        for p in _pats:
            for m in cwd.glob(p):
                self.remove(m, silent=not args.verbose)
            for m in cwd.glob("**/" + p):
                self.remove(m, silent=not args.verbose)


if __name__ == "__main__":
    Application().cmdline()
