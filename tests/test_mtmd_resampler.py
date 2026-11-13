"""Tests for MtmdResampler (mtmd_helper_resampler_*). No model required."""

import array
import math

import numpy as np
import pytest

from cyllama.llama.llama_cpp import MtmdResampler


def _sine(n, rate, freq=440.0):
    return array.array("f", (math.sin(2 * math.pi * freq * i / rate) for i in range(n)))


def test_equal_rates_pass_through():
    data = _sine(1000, 16000)
    with MtmdResampler(16000, 16000) as rs:
        out = rs.process(data)
    assert isinstance(out, array.array) and out.typecode == "f"
    assert out == data


def test_downsample_length():
    n = 48000
    with MtmdResampler(48000, 16000) as rs:
        out = rs.process(_sine(n, 48000))
    # linear resampler with filter latency: close to n/3, never more
    assert n // 3 - 64 <= len(out) <= n // 3


def test_chunked_matches_one_shot():
    data = np.frombuffer(_sine(44100, 44100), dtype=np.float32)
    with MtmdResampler(44100, 16000) as rs:
        whole = np.asarray(rs.process(data))
    with MtmdResampler(44100, 16000) as rs:
        parts = [np.asarray(rs.process(data[i : i + 441])) for i in range(0, len(data), 441)]
    chunked = np.concatenate(parts)
    assert len(chunked) == len(whole)
    np.testing.assert_allclose(chunked, whole, atol=1e-6)


def test_dc_signal_preserved():
    with MtmdResampler(44100, 16000) as rs:
        out = np.asarray(rs.process([0.5] * 44100))
    # past the filter warm-up, a low-pass of DC is DC
    np.testing.assert_allclose(out[len(out) // 2 :], 0.5, atol=1e-3)


def test_accepts_list_and_numpy():
    with MtmdResampler(32000, 16000) as rs:
        a = rs.process([0.0] * 320)
        b = rs.process(np.zeros(320, dtype=np.float32))
    assert len(a) > 0 and len(b) > 0


def test_empty_input():
    with MtmdResampler(48000, 16000) as rs:
        assert len(rs.process([])) == 0


@pytest.mark.parametrize("rates", [(0, 16000), (16000, 0), (-1, 16000)])
def test_invalid_rates(rates):
    with pytest.raises(ValueError):
        MtmdResampler(*rates)


def test_rates_exposed():
    rs = MtmdResampler(44100, 16000)
    assert (rs.rate_in, rs.rate_out) == (44100, 16000)
    rs.close()


def test_closed_raises_and_close_idempotent():
    rs = MtmdResampler(48000, 16000)
    rs.close()
    rs.close()
    with pytest.raises(RuntimeError, match="closed"):
        rs.process([0.0])
