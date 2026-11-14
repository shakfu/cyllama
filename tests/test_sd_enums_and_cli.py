"""SD enum coverage of upstream values, and the sd CLI's name parsing."""

import argparse

import pytest

pytest.importorskip("cyllama.sd")

from cyllama.sd import SDContextParams, SampleMethod, Scheduler, SDType
from cyllama.sd.__main__ import PREDICTION_CHOICES, parse_sampler_scheduler, prediction_map, vae_tile_kwargs
from cyllama.sd.stable_diffusion import Prediction, RngType, sample_method_name, scheduler_name, type_name


def test_default_sampler_rng_type_is_readable():
    # Upstream's default RNG_TYPE_COUNT means "same as rng_type".
    assert SDContextParams().sampler_rng_type is RngType.COUNT


@pytest.mark.parametrize("pred", [Prediction.FLUX3_FLOW, Prediction.H3_AV_FLOW, Prediction.MING_IMAGE_FLOW])
def test_new_prediction_types_roundtrip(pred):
    params = SDContextParams()
    params.prediction = pred
    assert params.prediction is pred


def test_prediction_count_follows_named_members():
    named = [p for p in Prediction if p is not Prediction.COUNT]
    assert Prediction.COUNT == len(named)


@pytest.mark.parametrize("t", [SDType.IQ4_NL, SDType.MXFP4, SDType.F8_E5M2])
def test_sd_type_roundtrip_and_name(t):
    params = SDContextParams()
    params.wtype = t
    assert params.wtype is t
    assert type_name(t) == t.name.lower()


def test_prediction_choices_match_map():
    assert PREDICTION_CHOICES == list(prediction_map())
    assert prediction_map()["sd3_flow"] is Prediction.FLOW


@pytest.mark.parametrize(
    "arg, expected",
    [
        ("dpm++2m", SampleMethod.DPMPP2M),
        ("dpmpp2m", SampleMethod.DPMPP2M),
        ("dpm++2mv2", SampleMethod.DPMPP2Mv2),
        ("dpmpp2mv2", SampleMethod.DPMPP2Mv2),
        ("Euler_A", SampleMethod.EULER_A),
    ],
)
def test_sampler_accepts_upstream_and_enum_names(arg, expected):
    method, _ = parse_sampler_scheduler(argparse.Namespace(sampler=arg, scheduler=None))
    assert method is expected


def test_every_upstream_sampler_and_scheduler_name_parses():
    for m in SampleMethod:
        if m is not SampleMethod.COUNT:
            assert parse_sampler_scheduler(argparse.Namespace(sampler=sample_method_name(m), scheduler=None))[0] is m
    for s in Scheduler:
        if s is not Scheduler.COUNT:
            assert parse_sampler_scheduler(argparse.Namespace(sampler=None, scheduler=scheduler_name(s)))[1] is s


def test_unknown_sampler_exits():
    with pytest.raises(SystemExit):
        parse_sampler_scheduler(argparse.Namespace(sampler="nope", scheduler=None))


def test_vae_tile_kwargs():
    assert vae_tile_kwargs(argparse.Namespace(vae_tile_size=None, vae_tile_overlap=None)) == {}
    assert vae_tile_kwargs(argparse.Namespace(vae_tile_size="512x384", vae_tile_overlap=0.25)) == {
        "vae_tile_size": (512, 384),
        "vae_tile_overlap": 0.25,
    }


@pytest.mark.parametrize("bad", ["512", "0x512", "ax512", "512x-1"])
def test_vae_tile_kwargs_rejects_malformed(bad):
    with pytest.raises(SystemExit):
        vae_tile_kwargs(argparse.Namespace(vae_tile_size=bad, vae_tile_overlap=None))
