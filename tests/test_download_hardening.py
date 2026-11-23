"""Model downloader hardening; local HTTP servers only, no network."""

import http.server
import os
import threading

import pytest

import cyllama.llama.llama_cpp as _cy


# ---------------------------------------------------------------------------
# Hardening: names from a remote manifest, endpoint scheme, redirects, resume
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["/etc/passwd", "../x.gguf", "a/../../x.gguf", "a\\..\\x.gguf", ""])
def test_manifest_file_name_confined_to_cache(tmp_path, name):
    with pytest.raises(ValueError, match="unsafe file name"):
        _cy._safe_cache_path(str(tmp_path), name)


def test_manifest_file_name_ok(tmp_path):
    assert _cy._safe_cache_path(str(tmp_path), "sub/model.gguf") == os.path.join(
        os.path.realpath(tmp_path), "sub", "model.gguf"
    )


@pytest.mark.parametrize("spec", ["user/../etc:latest", "user/repo:../x", "user/repo:a/b", "user/re po"])
def test_repo_and_tag_validated(spec):
    with pytest.raises(ValueError, match="invalid"):
        _cy._split_repo_tag(spec)


def test_repo_and_tag_ok():
    assert _cy._split_repo_tag("bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M") == (
        "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "Q4_K_M",
    )


def test_endpoint_scheme(monkeypatch):
    monkeypatch.setenv("LLAMA_CACHE_MODEL_ENDPOINT", "file:///etc")
    with pytest.raises(ValueError, match="http"):
        _cy._get_model_endpoint()
    monkeypatch.setenv("LLAMA_CACHE_MODEL_ENDPOINT", "http://mirror.example.com")
    with pytest.warns(UserWarning, match="plain http"):
        _cy._get_model_endpoint()


class _Recorder(http.server.BaseHTTPRequestHandler):
    seen = []
    redirect_to = None
    body = b"0123456789"
    etag = '"v2"'

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("ETag", self.etag)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()

    def do_GET(self):
        type(self).seen.append(dict(self.headers))
        if self.redirect_to:
            self.send_response(302)
            self.send_header("Location", self.redirect_to)
            self.end_headers()
            return
        # If-Range does not match: send the whole file
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, *a):
        pass


def _serve(**attrs):
    handler = type("H", (_Recorder,), {"seen": [], **attrs})
    srv = http.server.HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, handler


def test_authorization_dropped_on_cross_host_redirect():
    target, target_h = _serve()
    # 127.0.0.1 -> localhost: a different host name
    origin, _ = _serve(redirect_to=f"http://localhost:{target.server_address[1]}/f")
    try:
        status, _, body = _cy._url_request(
            f"http://127.0.0.1:{origin.server_address[1]}/f", headers={"Authorization": "Bearer secret"}
        )
        assert status == 200 and body == _Recorder.body
        assert all("Authorization" not in h for h in target_h.seen)
    finally:
        origin.shutdown()
        target.shutdown()


def test_resume_sends_if_range_and_restarts_on_change(tmp_path):
    srv, handler = _serve()
    try:
        dest = str(tmp_path / "m.gguf")
        with open(dest + ".downloadInProgress", "wb") as f:
            f.write(b"STALE")  # part of an older version
        assert _cy._download_file(f"http://127.0.0.1:{srv.server_address[1]}/m", dest)
        assert handler.seen[-1].get("If-Range") == '"v2"'
        with open(dest, "rb") as f:
            assert f.read() == _Recorder.body  # not b"STALE" + body
    finally:
        srv.shutdown()
