# Vulkan for Windows an MacOS intel

- **macOS Vulkan build failed with missing `libggml-cpu` then linker error** - Two issues: (1) `build_shared()` in `scripts/manage.py` only collected `**/*.dylib` on macOS, but `GGML_BACKEND_DL=ON` builds backends as CMake MODULE libraries which get `.so` extension on macOS — added `**/*.so` to the macOS glob patterns. (2) The linker rejected the collected `.so` files with "unsupported mach-o filetype" because MH_BUNDLE (MODULE) files cannot be linked, only `dlopen`'d at runtime — updated `_find_dylib` in `CMakeLists.txt` to detect `.so` files on macOS and treat them as install-only (shipped in the wheel for runtime loading, not passed to the linker)

```sh
$ git diff CMakeLists.txt
diff --git a/CMakeLists.txt b/CMakeLists.txt
index cdc38ba..851ee75 100644
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -305,7 +305,12 @@ if(WITH_DYLIB)
         list(APPEND _BACKEND_DYLIB_NAMES ggml-opencl)
     endif()

-    # Helper macro: find a shared lib, append to DYLIBS/DYLIB_FILES, or handle missing

+    # Helper macro: find a shared lib, append to DYLIBS/DYLIB_FILES, or handle missing.

+    # When _link_only is FALSE the lib is added to DYLIB_FILES (install) AND DYLIBS (link).

+    # When _link_only is FALSE and the file is a macOS MH_BUNDLE (.so), it is install-only:

+    # GGML_BACKEND_DL builds backends as CMake MODULE libraries which get a .so extension

+    # on macOS. These are dlopen'd at runtime by ggml and must NOT be passed to the linker

+    # (the linker rejects MH_BUNDLE with "unsupported mach-o filetype").
     macro(_find_dylib _name _is_required)
         find_library(_found_${_name}
             NAMES ${_name}
@@ -313,21 +318,30 @@ if(WITH_DYLIB)
             NO_DEFAULT_PATH
         )
         if(_found_${_name})
-            list(APPEND DYLIBS "${_found_${_name}}")

-            # Install the soname file -- that's what @rpath / $ORIGIN references at runtime

-            # macOS soname format: libllama.0.dylib

-            # Linux soname format: libllama.so.0

-            if(APPLE)

-                file(GLOB _soname "${LLAMACPP_DYLIB_DIR}/lib${_name}.[0-9].${_dylib_ext}")

-            else()

-                file(GLOB _soname "${LLAMACPP_DYLIB_DIR}/lib${_name}.${_dylib_ext}.[0-9]")

-            endif()

-            if(_soname)

-                list(APPEND DYLIB_FILES ${_soname})

-            else()

+            # Determine if this is a linkable shared lib or an install-only plugin.

+            # On macOS, .so files are MH_BUNDLE (MODULE) and cannot be linked.

+            get_filename_component(_ext "${_found_${_name}}" EXT)

+            if(APPLE AND "${_ext}" STREQUAL ".so")

+                # Install-only: ship in the wheel for runtime dlopen, don't link
                 list(APPEND DYLIB_FILES "${_found_${_name}}")

+                message(STATUS "  Found backend plugin (install-only): ${_found_${_name}}")

+            else()

+                list(APPEND DYLIBS "${_found_${_name}}")

+    # GGML_BACKEND_DL builds backends as CMake MODULE libraries which get a .s
o extension
+    # on macOS. These are dlopen'd at runtime by ggml and must NOT be passed t
o the linker
+    # (the linker rejects MH_BUNDLE with "unsupported mach-o filetype").
     macro(_find_dylib _name _is_required)
         find_library(_found_${_name}
             NAMES ${_name}
@@ -313,21 +318,30 @@ if(WITH_DYLIB)
             NO_DEFAULT_PATH
         )
         if(_found_${_name})
-            list(APPEND DYLIBS "${_found_${_name}}")

-            # Install the soname file -- that's what @rpath / $ORIGIN referenc
es at runtime
-            # macOS soname format: libllama.0.dylib

-            # Linux soname format: libllama.so.0

-            if(APPLE)

-                file(GLOB _soname "${LLAMACPP_DYLIB_DIR}/lib${_name}.[0-9].${_
dylib_ext}")
-            else()

-                file(GLOB _soname "${LLAMACPP_DYLIB_DIR}/lib${_name}.${_dylib_
ext}.[0-9]")
```
