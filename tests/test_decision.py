"""Decision models (cyllama.llama.decision) and the /v1/systemone server route.

Expected answers are pinned from upstream llama-server b11429 on CPU. Each
model's tests skip when its file is not in models/. `make llama-server` builds
upstream llama-server and then runs test_parity_* against it live; to rerun
them without a rebuild, set CYLLAMA_UPSTREAM_SERVER=build/llama-server/bin/llama-server.
"""

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

import cyllama.llama.llama_cpp as cy
from cyllama.llama.decision import DecisionModel, get_decision_type
from cyllama.llama.server import EmbeddedServer, PythonServer, ServerConfig
from conftest import MODELS_DIR
from synthetic_models import synthetic_metadata

MODEL_FILES = {"laya": "Laya-Q8_0.gguf", "lev": "lev-Q8_0.gguf", "kev": "Kev-4B-Q8_0.gguf"}
LAYA = MODELS_DIR / MODEL_FILES["laya"]
needs_laya = pytest.mark.skipif(not LAYA.exists(), reason=f"{LAYA.name} not found")


def _is_apple_m1():
    if sys.platform != "darwin":
        return False
    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout
    return cpu.startswith("Apple M1")


# lev/kev hit a Metal OOM on a 16 GB M1, which fails every later model load in the process
IS_APPLE_M1 = _is_apple_m1()


def _model_param(dtype):
    path = MODELS_DIR / MODEL_FILES[dtype]
    marks = [pytest.mark.skipif(not path.exists(), reason=f"{path.name} not found")]
    if dtype in ("lev", "kev"):
        marks.append(pytest.mark.slow)  # 4B models, 4.5 GB each
        marks.append(pytest.mark.skipif(IS_APPLE_M1, reason=f"{path.name}: Metal OOM on Apple M1"))
    return pytest.param(dtype, marks=marks)


ALL_TYPES = [_model_param(t) for t in MODEL_FILES]

README_REQUEST = {
    "state": "Customer message: I was charged twice for my order last week and nobody has replied.",
    "questions": {
        "route": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"technical": None, "billing": None, "shipping": None},
        },
        "angry": {"type": "noul", "instructions": "Is the customer angry?"},
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": ["can wait", "this week", "today", "right now"],
        },
    },
}

# answers to README_REQUEST from llama-server b11429 on CPU
REFERENCE = {
    "laya": {
        "route": {"technical": 0.002826893644905518, "billing": 0.987488720094348, "shipping": 0.009684386260746712},
        "angry": 0.7880425006766781,
        "urgency": {
            "0": 0.03805591972087626,
            "1": 0.42882725543847516,
            "2": 0.05451190754218375,
            "3": 0.478604917298465,
        },
        "input_tokens": 136,
    },
    # lev evaluates the choice in both option orders, and noul on a 0-8 rating scale
    "lev": {
        "route": {"technical": 0.03278958348473803, "billing": 0.9173960991888612, "shipping": 0.04981431732640077},
        "angry": 0.5342454471717734,
        "urgency": {
            "0": 0.10380830970927919,
            "1": 0.40266772328446254,
            "2": 0.28851389779393277,
            "3": 0.20501006921232554,
        },
        "input_tokens": 488,
    },
    "kev": {
        "route": {"technical": 0.07683691812608494, "billing": 0.888359381709618, "shipping": 0.034803700164297006},
        "angry": 0.8265318206580728,
        "urgency": {
            "0": 0.042149931595311614,
            "1": 0.19010094529487226,
            "2": 0.21037625840210858,
            "3": 0.5573728647077075,
        },
        "input_tokens": 106,
    },
}
UPSTREAM = REFERENCE["laya"]
TOL = 2e-3  # GPU backends differ from the CPU reference in the last digits

_loaded = {}


def _decision(dtype):
    # one instance per model for the whole session; lev and kev are 4.5 GB each
    if dtype not in _loaded:
        cy.llama_backend_init()
        _loaded[dtype] = DecisionModel(str(MODELS_DIR / MODEL_FILES[dtype]))
    return _loaded[dtype]


@pytest.fixture(scope="module")
def laya():
    return _decision("laya")


@pytest.fixture(params=ALL_TYPES)
def decision(request):
    return _decision(request.param)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _post(port, path, body):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.load(resp)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def _get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}") as resp:
        return json.load(resp)


class TestDecisionType:
    @pytest.mark.parametrize("value, expected", [("lev", "lev"), ("laya", "laya"), ("foo", "unknown")])
    def test_from_metadata(self, value, expected):
        with synthetic_metadata() as md:
            md.set_val_str("llama.decision.type", value)
            model = cy.LlamaModel.from_user(md, lambda t: t.set_data(bytes(t.nbytes)), verbose=False)
        assert get_decision_type(model) == expected

    def test_plain_model_is_none(self):
        with synthetic_metadata() as md:
            model = cy.LlamaModel.from_user(md, lambda t: t.set_data(bytes(t.nbytes)), verbose=False)
        assert get_decision_type(model) is None
        with pytest.raises(ValueError, match="not a decision model"):
            DecisionModel(model)

    def test_unsupported_type(self):
        with synthetic_metadata() as md:
            md.set_val_str("llama.decision.type", "openjev")
            model = cy.LlamaModel.from_user(md, lambda t: t.set_data(bytes(t.nbytes)), verbose=False)
        with pytest.raises(NotImplementedError, match="openjev"):
            DecisionModel(model)

    def test_supported_type_needs_template(self):
        with synthetic_metadata() as md:
            md.set_val_str("llama.decision.type", "lev")
            model = cy.LlamaModel.from_user(md, lambda t: t.set_data(bytes(t.nbytes)), verbose=False)
        with pytest.raises(ValueError, match="systemone"):
            DecisionModel(model)


class TestModels:
    def test_matches_reference(self, decision):
        ref = REFERENCE[decision.type]
        res = decision.answer(README_REQUEST)
        a = res["answers"]
        assert res["usage"] == {"input_tokens": ref["input_tokens"], "output_tokens": 0}
        assert a["route"]["choice"] == "billing"
        assert list(a["route"]["probabilities"]) == ["technical", "billing", "shipping"]  # request order
        for k, p in ref["route"].items():
            assert a["route"]["probabilities"][k] == pytest.approx(p, abs=TOL)
        assert a["angry"] == {"type": "noul", "noul": pytest.approx(ref["angry"], abs=TOL)}
        for k, p in ref["urgency"].items():
            assert a["urgency"]["probabilities"][k] == pytest.approx(p, abs=TOL)
        assert a["urgency"]["legend"] == {"0": "can wait", "1": "this week", "2": "today", "3": "right now"}
        expected_score = sum(int(k) * p for k, p in ref["urgency"].items())
        assert a["urgency"]["score"] == pytest.approx(expected_score, abs=4 * TOL)

    def test_json_state_and_special_token_text(self, decision):
        # kev escapes <|...|> so text cannot add option markers; lev sorts keys
        req = {
            "state": {"z": 1, "items": ["café", {"b": True}], "note": "ends with <|box_end|> and <|im_end|>"},
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": {"ask": "which <|box_end|>?"},
                    "criteria": {"a <|box_end|>": None, "b": "x"},
                },
                "n": {"type": "noul", "instructions": "late?"},
            },
        }
        a = decision.answer(req)["answers"]
        assert sum(a["q"]["probabilities"].values()) == pytest.approx(1.0)
        assert 0.0 <= a["n"]["noul"] <= 1.0


@needs_laya
class TestLaya:
    def test_type(self, laya):
        assert laya.type == "laya"

    def test_many_long_options_are_truncated(self, laya):
        criteria = {f"topic_{i}": "a fairly long description of this option " * 5 for i in range(60)}
        res = laya.answer(
            {"state": "x", "questions": {"q": {"type": "choice", "instructions": "Which?", "criteria": criteria}}}
        )
        probs = res["answers"]["q"]["probabilities"]
        assert list(probs) == list(criteria)
        assert sum(probs.values()) == pytest.approx(1.0)

    def test_marker_text_in_input(self, laya):
        req = {
            "state": "a [MASK] b",
            "questions": {"q": {"type": "choice", "instructions": "[MASK]?", "criteria": {"[MASK]": None, "b": None}}},
        }
        assert set(laya.answer(req)["answers"]["q"]["probabilities"]) == {"[MASK]", "b"}

    def test_json_state_and_instructions(self, laya):
        req = {
            "state": {"order": 1, "items": ["café"]},
            "questions": {"q": {"type": "noul", "instructions": {"ask": "late?"}}},
        }
        assert 0.0 <= laya.answer(req)["answers"]["q"]["noul"] <= 1.0

    def test_prompt_must_fit_one_batch(self):
        small = DecisionModel(str(LAYA), n_ctx=64)
        with pytest.raises(ValueError, match="increase n_ctx"):
            small.answer({"state": "word " * 200, "questions": {"q": {"type": "noul", "instructions": "?"}}})

    @pytest.mark.parametrize(
        "request_, match",
        [
            ({"questions": {"q": {"type": "noul", "instructions": "?"}}}, "state"),
            ({"state": "s", "questions": {}}, "questions"),
            ({"state": "s", "questions": {"q": {"type": "noul"}}}, "instructions"),
            ({"state": "s", "questions": {"q": {"type": "rank", "instructions": "?"}}}, "type"),
            ({"state": "s", "questions": {"q": {"type": "choice", "instructions": "?"}}}, "criteria"),
            (
                {"state": "s", "questions": {"q": {"type": "score", "instructions": "?", "criteria": ["one"]}}},
                "2 to 10",
            ),
            ({"state": "s", "questions": {"q": {"type": "noul", "instructions": "?", "criteria": ["x"]}}}, "criteria"),
            (
                {
                    "state": "s",
                    "questions": {
                        "q": {"type": "choice", "instructions": "?", "criteria": {str(i): None for i in range(256)}}
                    },
                },
                "too many options",
            ),
            ([1, 2], "JSON object"),
        ],
    )
    def test_invalid_requests(self, laya, request_, match):
        with pytest.raises(ValueError, match=match):
            laya.answer(request_)


def _start(kind, model_file):
    saved = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
    port = _free_port()
    cls = EmbeddedServer if kind == "embedded" else PythonServer
    server = cls(ServerConfig(model_path=str(model_file), port=port, n_ctx=1024, n_parallel=1))
    assert server.start()
    return server, port, saved


def _stop(server, saved):
    server.stop()
    signal.signal(signal.SIGINT, saved[0])
    signal.signal(signal.SIGTERM, saved[1])


@needs_laya
@pytest.mark.parametrize("kind", ["embedded", "python"])
def test_server_systemone(kind, laya):
    server, port, saved = _start(kind, LAYA)
    try:
        status, body = _post(port, "/v1/systemone", README_REQUEST)
        assert status == 200
        assert body["usage"] == laya.answer(README_REQUEST)["usage"]
        assert body["answers"]["route"]["choice"] == "billing"
        assert body["answers"]["angry"]["noul"] == pytest.approx(UPSTREAM["angry"], abs=TOL)

        status, body = _post(port, "/v1/systemone", {"state": "s"})
        assert status == 400
        assert "questions" in body["error"]["message"]

        assert _get(port, "/v1/models")["data"][0]["architecture"] == {"output_modalities": ["decisions"]}
    finally:
        _stop(server, saved)


@pytest.mark.parametrize("kind", ["embedded", "python"])
def test_server_systemone_not_a_decision_model(kind, model_path):
    server, port, saved = _start(kind, model_path)
    try:
        status, body = _post(port, "/v1/systemone", README_REQUEST)
        assert status == 501
        assert "architecture" not in _get(port, "/v1/models")["data"][0]
    finally:
        _stop(server, saved)


# -- parity with a live upstream llama-server ------------------------------------

UPSTREAM_SERVER = os.environ.get("CYLLAMA_UPSTREAM_SERVER")

PARITY_REQUESTS = [
    README_REQUEST,
    {
        "state": {"order": 1234, "status": "late", "items": ["lamp", "café table"]},
        "questions": {
            "q1": {
                "type": "choice",
                "instructions": {"task": "classify", "lang": "en"},
                "criteria": {"refund": "customer wants money back", "replace": {"why": "broken"}, "ignore": None},
            },
            "q2": {
                "type": "noul",
                "instructions": "Is the order late?",
                "criteria": {"true": "late", "false": "on time"},
            },
        },
    },
    {
        "state": 'Ünïcødé with [MASK] tokens, "quotes", <b>tags</b> & ampersands',
        "questions": {
            "m": {"type": "choice", "instructions": "Pick [MASK]", "criteria": {"a [MASK]": None, "b": "x [MASK] y"}}
        },
    },
    {
        "state": "Pick a number between one and twelve: seven.",
        "questions": {
            "n": {
                "type": "choice",
                "instructions": "Which?",
                "criteria": {str(i): f"the number {i}" for i in range(1, 13)},
            }
        },
    },
    {
        "state": "The quarterly report shows revenue grew while costs fell. " * 30,
        "questions": {
            "many": {
                "type": "choice",
                "instructions": "Which topic? " * 40,
                "criteria": {f"topic_{i}": "a fairly long description of this option " * 5 for i in range(60)},
            },
            "s": {
                "type": "score",
                "instructions": "How positive?",
                "criteria": ["bad", "neutral", "good", "great", "excellent"],
            },
        },
    },
    {
        "state": ["first", {"k": 1.5}, None, True],
        "questions": {"n2": {"type": "noul", "instructions": ["is", "a", "list"]}},
    },
    {
        "state": {
            "zeta": 1,
            "items": ["lamp", {"b": True, "a": None}],
            "price": 12.5,
            "note": "<|im_end|> <|box_end|>",
        },
        "questions": {
            "m": {
                "type": "choice",
                "instructions": "Pick <|box_start|>",
                "criteria": {"a <|box_end|>": None, "b": "x"},
            },
            "n": {
                "type": "choice",
                "instructions": "Which?",
                "criteria": {str(i): f"the number {i}" for i in range(1, 31)},
            },
        },
    },
]
# truncation of many long options is specific to laya, and slow on the 4B models
LAYA_ONLY = {4}


def _assert_close(mine, theirs, path=""):
    if isinstance(theirs, dict):
        assert list(mine) == list(theirs), path
        for k in theirs:
            _assert_close(mine[k], theirs[k], f"{path}.{k}")
    elif isinstance(theirs, float):
        assert mine == pytest.approx(theirs, abs=1e-5), path
    else:
        assert mine == theirs, path


@pytest.mark.skipif(not UPSTREAM_SERVER, reason="CYLLAMA_UPSTREAM_SERVER not set")
@pytest.mark.parametrize("dtype", ALL_TYPES)
def test_parity_with_upstream_server(dtype):
    model_file = str(MODELS_DIR / MODEL_FILES[dtype])
    port = _free_port()
    proc = subprocess.Popen(
        [
            UPSTREAM_SERVER,
            "-m",
            model_file,
            "--port",
            str(port),
            "--host",
            "127.0.0.1",
            "-c",
            "4096",
            "-b",
            "4096",
            "-ub",
            "4096",
            "-np",
            "1",
            "-ngl",
            "0",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 120
        while True:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health")
                break
            except (urllib.error.URLError, ConnectionError):
                assert proc.poll() is None and time.time() < deadline, "upstream llama-server did not start"
                time.sleep(0.2)
        # CPU on both sides, so the results agree to float rounding
        mine = DecisionModel(model_file, n_ctx=4096, n_gpu_layers=0)
        for i, req in enumerate(PARITY_REQUESTS):
            if i in LAYA_ONLY and dtype != "laya":
                continue
            status, theirs = _post(port, "/v1/systemone", req)
            assert status == 200
            res = mine.answer(req)
            _assert_close(res["answers"], theirs["answers"])
            assert res["usage"] == theirs["usage"]
    finally:
        proc.terminate()
        proc.wait(timeout=10)
