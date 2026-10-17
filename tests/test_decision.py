"""Decision models (cyllama.llama.decision) and the /v1/systemone server route.

Expected answers are pinned from upstream llama-server b11669 on CPU. Each
model's tests skip when its file is not in models/. `make llama-server` builds
upstream llama-server and then runs test_parity_* against it live; to rerun
them without a rebuild, set CYLLAMA_UPSTREAM_SERVER=build/llama-server/bin/llama-server.
"""

import json
import math
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
from cyllama.llama.decision import DecisionModel, get_decision_type, systemone_response
from cyllama.llama.server import EmbeddedServer, PythonServer, ServerConfig
from conftest import MODELS_DIR
from synthetic_models import save_synthetic_model, synthetic_metadata, synthetic_model

MODEL_FILES = {
    "laya": "Laya-Q8_0.gguf",
    "lev": "lev-Q8_0.gguf",
    "kev": "Kev-4B-Q8_0.gguf",
    "lfm2-d1-omni": "d1-omni-600M-Q8_0.gguf",
}
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

# answers to README_REQUEST from llama-server b11669 on CPU
REFERENCE = {
    "laya": {
        "route": {"technical": 0.002645999136237623, "billing": 0.9877295044860257, "shipping": 0.00962449637773663},
        "angry": 0.7870874820229915,
        "urgency": {
            "0": 0.03708144101697737,
            "1": 0.46304134918865114,
            "2": 0.049762526823041595,
            "3": 0.45011468297132995,
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
    "lfm2-d1-omni": {
        "route": {"technical": 0.0012363145224966168, "billing": 0.9987139813165001, "shipping": 4.970416100308494e-05},
        "angry": 0.3794288830523013,
        "urgency": {
            "0": 0.005675338866501018,
            "1": 0.9865855906695629,
            "2": 0.003307158356302457,
            "3": 0.004431912107633534,
        },
        "input_tokens": 152,
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


# random-weight models with the "test" tokenizer, for the types without a downloaded model
SYNTHETIC_TYPES = ("pplx-decider", "vllm-d3", "lfm2-d1")
SYNTHETIC_TEMPLATE = (
    "{{ state if state is string else state | tojson }}\n"
    "{{ instructions if instructions is string else instructions | tojson }}\n"
    "{% for o in options %}{{ o.label }}: {{ o.key }}"
    "{% if o.description is not none %} - {{ o.description if o.description is string else o.description | tojson }}"
    "{% endif %}\n{% endfor %}Answer:"
)
SCORE_BIAS = {"score_bias.4.1": 2.0, "score_bias.4.3": -1.5}


def _synthetic_extra(dtype, **decision):
    extra = {"llama.decision.type": dtype, "tokenizer.chat_template.systemone": SYNTHETIC_TEMPLATE}
    extra.update({f"llama.decision.{k}": v for k, v in decision.items()})
    return extra


def _synthetic_decision(dtype, cache_prompt=True, **decision):
    model = synthetic_model(stdev=0.5, n_ctx=4096, extra=_synthetic_extra(dtype, **decision))
    return DecisionModel(model, n_ctx=4096, n_gpu_layers=0, cache_prompt=cache_prompt)


@pytest.fixture(scope="module")
def synthetic_file(tmp_path_factory):
    """Path of a GGUF file of a synthetic decision model, with a score bias, by type."""
    root = tmp_path_factory.mktemp("decision")

    def make(dtype):
        path = root / f"{dtype}.gguf"
        if not path.exists():
            save_synthetic_model(path, stdev=0.5, n_ctx=4096, extra=_synthetic_extra(dtype, **SCORE_BIAS))
        return path

    return make


class TestSynthetic:
    @pytest.mark.parametrize("dtype", SYNTHETIC_TYPES)
    def test_answers(self, dtype):
        a = _synthetic_decision(dtype).answer(README_REQUEST)["answers"]
        assert list(a["route"]["probabilities"]) == ["technical", "billing", "shipping"]
        assert sum(a["route"]["probabilities"].values()) == pytest.approx(1.0)
        assert sum(a["urgency"]["probabilities"].values()) == pytest.approx(1.0)
        assert 0.0 <= a["angry"]["noul"] <= 1.0

    def test_score_bias(self):
        # a bias b_i multiplies the probability of level i by exp(b_i / T), T = 1 here
        plain = _synthetic_decision("pplx-decider").answer(README_REQUEST)["answers"]
        biased = _synthetic_decision("pplx-decider", **SCORE_BIAS).answer(README_REQUEST)["answers"]
        bias = [0.0, 2.0, 0.0, -1.5]
        weights = [p * math.exp(b) for p, b in zip(plain["urgency"]["probabilities"].values(), bias)]
        expected = [w / sum(weights) for w in weights]
        assert list(biased["urgency"]["probabilities"].values()) == pytest.approx(expected, abs=1e-9)
        assert biased["route"] == plain["route"]  # score questions only

    def test_score_bias_of_other_level_count_is_ignored(self):
        plain = _synthetic_decision("pplx-decider").answer(README_REQUEST)["answers"]
        other = _synthetic_decision("pplx-decider", **{"score_bias.3.0": 5.0}).answer(README_REQUEST)["answers"]
        assert other["urgency"] == plain["urgency"]

    @pytest.mark.parametrize("key", ["score_bias.4.4", "score_bias.1.0", "score_bias.4", "score_bias.x.0"])
    def test_invalid_score_bias(self, key):
        with pytest.raises(ValueError, match="score bias"):
            _synthetic_decision("pplx-decider", **{key: 1.0})

    @pytest.mark.parametrize("dtype, ok", [("vllm-d3", True), ("lfm2-d1", True), ("pplx-decider", False)])
    def test_null_state(self, dtype, ok):
        dm = _synthetic_decision(dtype)
        req = {"state": None, "questions": {"q": {"type": "noul", "instructions": "?"}}}
        if ok:
            assert 0.0 <= dm.answer(req)["answers"]["q"]["noul"] <= 1.0
        else:
            with pytest.raises(ValueError, match="state"):
                dm.answer(req)
        with pytest.raises(ValueError, match="state"):
            dm.answer({"questions": req["questions"]})

    def test_vllm_d3(self):
        dm = _synthetic_decision("vllm-d3")
        a = dm.answer(
            {
                "state": "s",
                "questions": {
                    "q": {"type": "score", "criteria": [{"b": 1, "a": "é"}, None, "plain"]}
                },  # no instructions
            }
        )["answers"]["q"]
        assert a["legend"] == {"0": '{"a":"é","b":1}', "1": "null", "2": "plain"}
        probs = list(a["probabilities"].values())
        entropy = -sum(p * math.log(p) for p in probs)
        assert a["confidence"] == pytest.approx(1.0 - entropy / math.log(3))
        with pytest.raises(ValueError, match="instructions"):
            _synthetic_decision("pplx-decider").answer({"state": "s", "questions": {"q": {"type": "noul"}}})

    def test_lfm2_d1_labels(self):
        dm = _synthetic_decision("lfm2-d1")
        texts, groups = dm._d1_labels({"type": "choice", "options": [("x", None), ("y", None)]})
        assert texts == ["x", "y"]  # single letters are their own labels
        texts, _ = dm._d1_labels({"type": "choice", "options": [("one", None), ("two", None)]})
        assert texts == ["A", "B"]
        firsts = [g[0] for g in groups]
        assert len(set(firsts)) == len(firsts)

    def test_cache_prompt(self, monkeypatch):
        long_req = PARITY_REQUESTS[4]
        req = {"state": long_req["state"], "questions": {"s": long_req["questions"]["s"]}}
        cold = _synthetic_decision("pplx-decider", cache_prompt=False)
        first = cold.answer(req)
        cold.answer(long_req)
        assert cold.answer(req) == first  # repeatable

        warm = _synthetic_decision("pplx-decider")
        n_past = []
        decode = warm._decode
        monkeypatch.setattr(warm, "_decode", lambda *a, **kw: n_past.append(decode(*a, **kw)) or n_past[-1])
        assert warm.answer(req) == first  # nothing cached yet
        warm.answer(long_req)
        warm.answer(req)
        assert n_past[0] == 0 and min(n_past[1:]) > 0  # the prompts start with the same state

    @pytest.mark.parametrize(
        "extra",
        [
            {"images": ["data:image/png;base64,AAAA"]},
            {"files": ["data:audio/wav;base64,AAAA"]},
            {"state": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:,"}}]}]},
            {"state": {"messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {}}]}]}},
        ],
    )
    def test_media_is_refused(self, extra):
        dm = _synthetic_decision("pplx-decider")
        req = {**README_REQUEST, **extra}
        with pytest.raises(NotImplementedError, match="not supported"):
            dm.answer(req)
        status, body = systemone_response(dm, req, "m")
        assert status == 501 and body["error"]["type"] == "not_supported_error"

    def test_media_errors(self):
        dm = _synthetic_decision("pplx-decider")
        for extra, match in [({"videos": ["x"]}, "videos"), ({"images": "x"}, "array")]:
            with pytest.raises(ValueError, match=match):
                dm.answer({**README_REQUEST, **extra})
        dm.answer({**README_REQUEST, "images": [], "videos": None})


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
# many long options: cut to fit by laya and lfm2-d1-omni, slow on the 4B models
SKIP_4B = {4}


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
@pytest.mark.parametrize("dtype", ALL_TYPES + list(SYNTHETIC_TYPES))
def test_parity_with_upstream_server(dtype, synthetic_file):
    model_file = str(synthetic_file(dtype) if dtype in SYNTHETIC_TYPES else MODELS_DIR / MODEL_FILES[dtype])
    mine = DecisionModel(model_file, n_ctx=4096, n_gpu_layers=0)
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
            "-ub",  # the ubatch split changes the results by more than float rounding
            str(mine.ctx.n_ubatch),
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
        # CPU on both sides, so the results agree to float rounding. Both reuse the KV
        # prefix of the last prompt, so the requests go in the same order to both
        for i, req in enumerate(PARITY_REQUESTS):
            if i in SKIP_4B and dtype in ("lev", "kev"):
                continue
            status, theirs = _post(port, "/v1/systemone", req)
            assert status == 200
            res = mine.answer(req)
            _assert_close(res["answers"], theirs["answers"])
            assert res["usage"] == theirs["usage"]
    finally:
        proc.terminate()
        proc.wait(timeout=10)
