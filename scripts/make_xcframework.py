#!/usr/bin/env python3
"""make_xcframework.py: build four xcframeworks on macOS arm64.

Produces:
    dist/Ggml.xcframework             (libggml{,-base,-cpu,-metal})
    dist/LlamaCpp.xcframework         (libllama, libmtmd) -> Ggml
    dist/Whisper.xcframework          (libwhisper)        -> Ggml
    dist/StableDiffusion.xcframework  (libstable-diffusion) -> Ggml

All four share one set of ggml dylibs (Metal + CPU) sourced from llama.cpp.
Each framework ships an umbrella binary that re-exports its component libs,
so a consumer can `-framework LlamaCpp -framework Ggml` and pick up every
public symbol.

Install-name scheme:
    @rpath/<Component>.framework/Versions/A/Libraries/<basename>.dylib
    @rpath/<Component>.framework/Versions/A/<Component>             (umbrella)

Consumers add ONE rpath (e.g. @executable_path/../Frameworks) pointing at
the directory that holds all four .framework bundles as siblings, and all
inter-framework references resolve.

Run:  make xcframework      (or)  python scripts/make_xcframework.py
"""
from __future__ import annotations

import os
import plistlib
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths and constants

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
STAGE = ROOT / "build" / "xcframework"

FRAMEWORK_VERSION = "A"
BUNDLE_VERSION = "1"
SHORT_VERSION = "0.1.0"
MIN_MACOS = os.environ.get("MACOSX_DEPLOYMENT_TARGET", "14.0")

LLAMA_DIR = ROOT / "thirdparty" / "llama.cpp"
WHISPER_DIR = ROOT / "thirdparty" / "whisper.cpp"
SD_DIR = ROOT / "thirdparty" / "stable-diffusion.cpp"

LLAMA_DYN = LLAMA_DIR / "dynamic"
WHISPER_DYN = WHISPER_DIR / "dynamic"
SD_DYN = SD_DIR / "dynamic"

LLAMA_SRC = ROOT / "build" / "llama.cpp"
WHISPER_SRC = ROOT / "build" / "whisper.cpp"
SD_SRC = ROOT / "build" / "stable-diffusion.cpp"


# ---------------------------------------------------------------------------
# Component model

@dataclass
class Component:
    name: str                       # bundle name, e.g. "LlamaCpp"
    bundle_id: str
    src_dyn_dir: Path               # where to pick the dylibs from
    lib_stems: list[str]            # bare names without ".dylib"
    header_sources: list[tuple[Path, list[str]]]  # (include_dir, [filenames])
    deps: list[str] = field(default_factory=list)  # other Component.name strings


COMPONENTS: list[Component] = [
    Component(
        name="Ggml",
        bundle_id="com.cyllama.ggml",
        src_dyn_dir=LLAMA_DYN,  # canonical ggml from llama.cpp
        lib_stems=["libggml", "libggml-base", "libggml-cpu",
                   "libggml-metal", "libggml-blas"],
        header_sources=[(LLAMA_DIR / "include", [
            "ggml.h", "ggml-alloc.h", "ggml-backend.h", "ggml-blas.h",
            "ggml-cpp.h", "ggml-cpu.h", "ggml-metal.h", "ggml-opt.h",
            "gguf.h",
        ])],
    ),
    Component(
        name="LlamaCpp",
        bundle_id="com.cyllama.llamacpp",
        src_dyn_dir=LLAMA_DYN,
        lib_stems=["libllama", "libmtmd"],
        header_sources=[(LLAMA_DIR / "include", [
            "llama.h", "llama-cpp.h", "mtmd.h", "mtmd-helper.h",
        ])],
        deps=["Ggml"],
    ),
    Component(
        name="Whisper",
        bundle_id="com.cyllama.whisper",
        src_dyn_dir=WHISPER_DYN,
        lib_stems=["libwhisper"],
        header_sources=[(WHISPER_DIR / "include", ["whisper.h"])],
        deps=["Ggml"],
    ),
    Component(
        name="StableDiffusion",
        bundle_id="com.cyllama.stablediffusion",
        src_dyn_dir=SD_DYN,
        lib_stems=["libstable-diffusion"],
        header_sources=[(SD_DIR / "include", ["stable-diffusion.h"])],
        deps=["Ggml"],
    ),
]


# Map each library basename to the framework that owns it. Used when
# rewriting LC_LOAD_DYLIB entries: any reference whose basename matches
# (after soname stripping) gets rewritten to the canonical install name
# of the owning framework.
def _owner_map() -> dict[str, Component]:
    m: dict[str, Component] = {}
    for c in COMPONENTS:
        for stem in c.lib_stems:
            m[f"{stem}.dylib"] = c
    return m


# ---------------------------------------------------------------------------
# Shell helpers

def run(cmd, cwd=None, env=None):
    print(f"+ {' '.join(str(c) for c in cmd)}")
    subprocess.run([str(c) for c in cmd], check=True, cwd=cwd, env=env)


def run_capture(cmd) -> str:
    return subprocess.run(
        [str(c) for c in cmd], check=True, capture_output=True, text=True
    ).stdout


def fail(msg):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Step 1: ensure shared dylibs exist for all three projects

def ensure_dylibs():
    """Build llama/whisper/SD as shared libs sharing llama.cpp's ggml.

    Idempotent: skips any project whose dylib is already present.
    """
    env = {**os.environ, "SD_USE_VENDORED_GGML": "0"}

    # Ensure source trees are cloned. manage.py's --deps-only path runs
    # the static build but more importantly clones the repos.
    if not LLAMA_SRC.exists():
        run([sys.executable, "scripts/manage.py", "build",
             "--llama-cpp", "--deps-only"], cwd=ROOT, env=env)
    if not WHISPER_SRC.exists():
        run([sys.executable, "scripts/manage.py", "build",
             "--whisper-cpp", "--deps-only"], cwd=ROOT, env=env)
    if not SD_SRC.exists():
        run([sys.executable, "scripts/manage.py", "build",
             "--stable-diffusion", "--deps-only", "--sd-shared-ggml",
             "--no-sd-examples"], cwd=ROOT, env=env)

    # Build llama.cpp shared with GGML_BACKEND_DL=OFF so every ggml backend
    # is a real MH_DYLIB (not MH_BUNDLE), which is required for the umbrella
    # binary's -reexport_library link step.
    if not (LLAMA_DYN / "libllama.dylib").exists():
        _build_shared_cmake(
            src=LLAMA_SRC,
            targets=["llama", "mtmd", "ggml", "ggml-base",
                     "ggml-cpu", "ggml-metal", "ggml-blas"],
            dst=LLAMA_DYN,
            extra_cmake=[
                "-DGGML_METAL=ON",
                "-DGGML_METAL_EMBED_LIBRARY=ON",
                "-DGGML_BLAS=ON",
                "-DGGML_BACKEND_DL=OFF",
                "-DLLAMA_CURL=OFF",
                "-DLLAMA_BUILD_SERVER=OFF",
                "-DLLAMA_BUILD_TESTS=OFF",
                "-DLLAMA_BUILD_EXAMPLES=OFF",
            ],
            sync_ggml_from=None,
            collect_globs=["**/libllama*.dylib", "**/libmtmd*.dylib",
                           "**/libggml*.dylib"],
            require=["libllama.dylib", "libmtmd.dylib",
                     "libggml.dylib", "libggml-base.dylib",
                     "libggml-cpu.dylib", "libggml-metal.dylib",
                     "libggml-blas.dylib"],
        )

    if not (WHISPER_DYN / "libwhisper.dylib").exists():
        _build_shared_cmake(
            src=WHISPER_SRC,
            targets=["whisper"],
            dst=WHISPER_DYN,
            extra_cmake=[
                "-DGGML_METAL=ON",
                "-DGGML_METAL_EMBED_LIBRARY=ON",
                "-DGGML_BACKEND_DL=OFF",
                "-DWHISPER_BUILD_TESTS=OFF",
                "-DWHISPER_BUILD_EXAMPLES=OFF",
            ],
            sync_ggml_from=LLAMA_SRC / "ggml",
            collect_globs=["**/libwhisper*.dylib"],
            require=["libwhisper.dylib"],
        )

    if not (SD_DYN / "libstable-diffusion.dylib").exists():
        _build_shared_cmake(
            src=SD_SRC,
            targets=["stable-diffusion"],
            dst=SD_DYN,
            extra_cmake=[
                "-DSD_METAL=ON",
                "-DSD_BUILD_SHARED_LIBS=ON",
                "-DSD_BUILD_SHARED_GGML_LIB=ON",
                "-DSD_BUILD_EXAMPLES=OFF",
                "-DGGML_METAL_EMBED_LIBRARY=ON",
                "-DGGML_BACKEND_DL=OFF",
            ],
            sync_ggml_from=LLAMA_SRC / "ggml",
            collect_globs=["**/libstable-diffusion*.dylib"],
            require=["libstable-diffusion.dylib"],
        )


def _build_shared_cmake(src, targets, dst, extra_cmake, sync_ggml_from,
                        collect_globs, require):
    """Run a fresh cmake build with BUILD_SHARED_LIBS=ON and collect dylibs.

    `targets` is a list of cmake target names to build.
    `require` lists basenames that MUST appear in `dst` after collection;
    the function fails loudly otherwise (catches silent static-lib builds).
    """
    if sync_ggml_from and sync_ggml_from.exists() and (src / "ggml").exists():
        print(f"syncing ggml: {sync_ggml_from} -> {src/'ggml'}")
        shutil.rmtree(src / "ggml")
        shutil.copytree(sync_ggml_from, src / "ggml")

    bld = src / "build_shared"
    if bld.exists():
        shutil.rmtree(bld)
    bld.mkdir(parents=True)

    cmake_cmd = [
        "cmake", "-S", src, "-B", bld,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=ON",
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
        "-DGGML_NATIVE=OFF",
        "-DCMAKE_C_FLAGS=-DGGML_MAX_NAME=128",
        "-DCMAKE_CXX_FLAGS=-DGGML_MAX_NAME=128",
        f"-DCMAKE_OSX_DEPLOYMENT_TARGET={MIN_MACOS}",
        *extra_cmake,
    ]
    run(cmake_cmd)
    build_cmd = ["cmake", "--build", bld, "--config", "Release", "-j"]
    for t in targets:
        build_cmd += ["--target", t]
    run(build_cmd)

    dst.mkdir(parents=True, exist_ok=True)
    collected: set[str] = set()
    for pattern in collect_globs:
        # Resolve unique real files (skip symlinks pointing to versioned names)
        seen: set[Path] = set()
        for f in bld.glob(pattern):
            real = f.resolve()
            if real in seen or not real.is_file():
                continue
            seen.add(real)
            # Strip soname: libfoo.0.1.2.dylib -> libfoo.dylib
            target_name = _strip_soname(f.name)
            shutil.copy2(real, dst / target_name)
            collected.add(target_name)
            print(f"  collected {target_name}")

    missing = [n for n in require if n not in collected]
    if missing:
        fail(f"shared build of {src.name} failed to produce: {missing}. "
             f"Built artifacts in {bld}.")


# ---------------------------------------------------------------------------
# Step 2: stage one Cyllama-internal framework per component

def install_name_for(component: Component, basename: str) -> str:
    """Canonical @rpath path for a library inside its owning framework."""
    return (
        f"@rpath/{component.name}.framework/Versions/{FRAMEWORK_VERSION}/"
        f"Libraries/{basename}"
    )


def umbrella_install_name(component: Component) -> str:
    return (
        f"@rpath/{component.name}.framework/Versions/{FRAMEWORK_VERSION}/"
        f"{component.name}"
    )


def stage_framework(component: Component, owners: dict[str, Component]) -> Path:
    fw = STAGE / f"{component.name}.framework"
    if fw.exists():
        shutil.rmtree(fw)

    versioned = fw / "Versions" / FRAMEWORK_VERSION
    libs = versioned / "Libraries"
    headers = versioned / "Headers"
    modules = versioned / "Modules"
    resources = versioned / "Resources"
    for d in (libs, headers, modules, resources):
        d.mkdir(parents=True)

    _copy_resolved(component.src_dyn_dir, component.lib_stems, libs)
    _normalize_libs(component, libs, owners)
    _copy_headers(component, headers)
    _patch_cross_framework_includes(component, headers)

    umbrella = versioned / component.name
    _build_umbrella(component, umbrella, libs)

    _write_info_plist(resources / "Info.plist", component)
    _write_module_map(modules / "module.modulemap", component, headers)
    _make_version_symlinks(fw, component)
    return fw


def _copy_resolved(src_dir: Path, stems: list[str], dst_dir: Path):
    """Copy each <stem>.dylib from src_dir to dst_dir, resolving symlinks
    so we get a single real file per name (no versioned soname duplicates)."""
    for stem in stems:
        src = src_dir / f"{stem}.dylib"
        if not src.exists():
            fail(f"missing dylib: {src}")
        real = src.resolve()
        dst = dst_dir / f"{stem}.dylib"
        shutil.copy2(real, dst)
        os.chmod(dst, 0o755)
        print(f"  staged {dst.name} -> {component_short(dst)}")


def component_short(p: Path) -> str:
    """Pretty path relative to STAGE for log lines."""
    try:
        return str(p.relative_to(STAGE))
    except ValueError:
        return str(p)


def _normalize_libs(component: Component, libs_dir: Path,
                    owners: dict[str, Component]):
    """Set install names + rewrite LC_LOAD_DYLIB so every reference uses
    the canonical @rpath/<Owner>.framework/Versions/A/Libraries/<name> form,
    and replace rpaths with @loader_path/../../../.. so @rpath/<Owner>...
    resolves at the directory holding all .framework bundles."""
    for f in sorted(libs_dir.glob("*.dylib")):
        # 1) install id
        run(["install_name_tool", "-id",
             install_name_for(component, f.name), str(f)])

        # 2) rewrite each LC_LOAD_DYLIB whose basename we own
        otool = run_capture(["otool", "-L", str(f)])
        for line in otool.splitlines()[1:]:
            line = line.strip()
            if not line:
                continue
            old = line.split(" (", 1)[0].strip()
            base = Path(old).name
            stripped = _strip_soname(base)
            owner = owners.get(stripped) or owners.get(base)
            if owner is None:
                continue
            new = install_name_for(owner, stripped)
            if new != old:
                run(["install_name_tool", "-change", old, new, str(f)])

        # 3) reset rpaths: one entry pointing at the directory containing
        #    all .framework bundles (relative to the lib's location).
        for rp in _existing_rpaths(f):
            subprocess.run(
                ["install_name_tool", "-delete_rpath", rp, str(f)],
                check=False, capture_output=True,
            )
        run(["install_name_tool", "-add_rpath",
             "@loader_path/../../../..", str(f)])


def _strip_soname(name: str) -> str:
    """libggml.0.dylib -> libggml.dylib; libfoo.1.2.3.dylib -> libfoo.dylib."""
    if not name.endswith(".dylib"):
        return name
    stem = name[: -len(".dylib")]
    parts = stem.split(".")
    while len(parts) > 1 and parts[-1].isdigit():
        parts.pop()
    return ".".join(parts) + ".dylib"


def _existing_rpaths(dylib: Path) -> list[str]:
    out = run_capture(["otool", "-l", str(dylib)])
    rpaths: list[str] = []
    lines = out.splitlines()
    for i, line in enumerate(lines):
        if "cmd LC_RPATH" not in line:
            continue
        for j in range(i + 1, min(i + 4, len(lines))):
            if "path " in lines[j]:
                seg = lines[j].split("path ", 1)[1]
                rpaths.append(seg.split(" (offset", 1)[0].strip())
                break
    return rpaths


def _copy_headers(component: Component, dst: Path):
    for inc, names in component.header_sources:
        for name in names:
            src = inc / name
            if not src.exists():
                print(f"  warn: missing header {src}")
                continue
            shutil.copy2(src, dst / name)
            print(f"  header {component.name}/{name}")


def _header_owner_map() -> dict[str, Component]:
    """Map every shipped header filename to its owning component."""
    m: dict[str, Component] = {}
    for c in COMPONENTS:
        for _inc, names in c.header_sources:
            for n in names:
                m[n] = c
    return m


_INCLUDE_RE = re.compile(r'^(\s*#\s*include\s*)"([^"]+)"', re.MULTILINE)


def _patch_cross_framework_includes(component: Component, headers_dir: Path):
    """Rewrite quoted #include directives that reference headers owned by
    another component, into Apple framework form `<Owner/header.h>`.

    Example: in LlamaCpp/llama.h, `#include "ggml.h"` -> `#include <Ggml/ggml.h>`
    so Swift's `import LlamaCpp` (which builds the Clang module) can resolve
    the cross-framework reference via the standard framework search path
    without consumers adding extra -I flags.
    """
    owners = _header_owner_map()
    for hdr in sorted(headers_dir.glob("*.h")):
        text = hdr.read_text()
        changed = False

        def repl(match):
            nonlocal changed
            prefix, included = match.group(1), match.group(2)
            base = Path(included).name
            owner = owners.get(base)
            if owner is None or owner.name == component.name:
                return match.group(0)
            changed = True
            return f"{prefix}<{owner.name}/{base}>"

        new_text = _INCLUDE_RE.sub(repl, text)
        if changed:
            hdr.write_text(new_text)
            print(f"  patched cross-framework includes in {component.name}/{hdr.name}")


def _build_umbrella(component: Component, out: Path, libs_dir: Path):
    """Build a thin dylib that re-exports the component's public libraries.

    Re-export is recorded against each dependency's *current* install name,
    so we run this AFTER _normalize_libs has set the canonical @rpath ids.
    """
    stub_c = STAGE / f"_{component.name.lower()}_umbrella.c"
    stub_c.write_text(
        f"void {component.name.lower()}_umbrella_anchor(void) {{}}\n"
    )

    cmd = [
        "clang", "-dynamiclib",
        f"-mmacosx-version-min={MIN_MACOS}",
        "-o", str(out),
        str(stub_c),
        "-install_name", umbrella_install_name(component),
    ]
    for stem in component.lib_stems:
        cmd += ["-Wl,-reexport_library," + str(libs_dir / f"{stem}.dylib")]
    # Umbrella binary lives at Versions/A/<Name>; @loader_path/../../..
    # is the directory containing all .framework bundles.
    cmd += ["-Wl,-rpath,@loader_path/../../.."]
    run(cmd)
    os.chmod(out, 0o755)


def _write_info_plist(path: Path, component: Component):
    plist = {
        "CFBundleDevelopmentRegion": "en",
        "CFBundleExecutable": component.name,
        "CFBundleIdentifier": component.bundle_id,
        "CFBundleInfoDictionaryVersion": "6.0",
        "CFBundleName": component.name,
        "CFBundlePackageType": "FMWK",
        "CFBundleShortVersionString": SHORT_VERSION,
        "CFBundleSignature": "????",
        "CFBundleVersion": BUNDLE_VERSION,
        "LSMinimumSystemVersion": MIN_MACOS,
    }
    with path.open("wb") as f:
        plistlib.dump(plist, f)


def _write_module_map(path: Path, component: Component, headers_dir: Path):
    c_headers: list[str] = []
    cpp_headers: list[str] = []
    for h in sorted(headers_dir.glob("*.h")):
        # Headers ending in -cpp.h (e.g. llama-cpp.h, ggml-cpp.h) gate
        # themselves with `#error "This header is for C++ only"`. Put them
        # behind `requires cplusplus` so C consumers don't pull them in.
        if h.stem.endswith("-cpp"):
            cpp_headers.append(h.name)
        else:
            c_headers.append(h.name)

    lines = [f"framework module {component.name} {{"]
    for h in c_headers:
        lines.append(f'    header "{h}"')
    lines.append("    export *")
    for dep in component.deps:
        lines.append(f"    use {dep}")
    if cpp_headers:
        lines.append(f"    explicit module Cpp {{")
        lines.append("        requires cplusplus")
        for h in cpp_headers:
            lines.append(f'        header "{h}"')
        lines.append("        export *")
        lines.append("    }")
    lines.append("}")
    path.write_text("\n".join(lines) + "\n")


def _make_version_symlinks(fw: Path, component: Component):
    versions = fw / "Versions"
    current = versions / "Current"
    if current.exists() or current.is_symlink():
        current.unlink()
    current.symlink_to(FRAMEWORK_VERSION)

    for entry in (component.name, "Headers", "Modules", "Resources", "Libraries"):
        link = fw / entry
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(f"Versions/Current/{entry}")


# ---------------------------------------------------------------------------
# Step 3: xcodebuild -create-xcframework

def create_xcframework(framework: Path, name: str) -> Path:
    DIST.mkdir(parents=True, exist_ok=True)
    out = DIST / f"{name}.xcframework"
    if out.exists():
        shutil.rmtree(out)
    run([
        "xcodebuild", "-create-xcframework",
        "-framework", str(framework),
        "-output", str(out),
    ])
    return out


# ---------------------------------------------------------------------------

def main():
    if sys.platform != "darwin":
        fail("xcframework target is macOS-only")

    STAGE.mkdir(parents=True, exist_ok=True)
    ensure_dylibs()

    owners = _owner_map()
    built: list[Path] = []
    for component in COMPONENTS:
        print(f"\n=== staging {component.name}.framework ===")
        fw = stage_framework(component, owners)
        print(f"\n=== creating {component.name}.xcframework ===")
        built.append(create_xcframework(fw, component.name))

    print("\nbuilt:")
    for p in built:
        print(f"  {p}")


if __name__ == "__main__":
    main()
