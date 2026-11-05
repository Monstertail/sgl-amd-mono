# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3.1's spec is Kimi-K3's under K3.1's admission."""

import sgl_amd_mono
from sgl_amd_mono.kimi_k3 import spec as k3
from sgl_amd_mono.kimi_k31 import spec as k31

try:
    from sglang.kernels.plugin_kernels.amd.spec import Feature
except ModuleNotFoundError:
    from sglang.srt.monokernels.spec import Feature


def test_registered_under_the_k31_architecture():
    assert (
        sgl_amd_mono.SPECS[("KimiK31ForConditionalGeneration", None)]
        == "sgl_amd_mono.kimi_k31.spec:SPEC"
    )


def test_admits_the_k31_recipe():
    req = k31.SPEC.requires
    assert k31.SPEC.name == "kimi_k31_hip"
    assert req.hf_config == {"model_type": "kimi_k31"}
    assert "DFLASH" in req.server_args["speculative_algorithm"]
    assert {"aiter", "dsa"} <= req.attention_backends
    assert {Feature.KV_OFFLOAD, Feature.CONTEXT_PARALLEL} <= k31.SPEC.features


def test_shares_the_k3_kernels_and_checks():
    for field in (
        "devices",
        "numerics",
        "build",
        "model_disable_reasons",
        "config_disable_reasons",
    ):
        assert getattr(k31.SPEC, field) is getattr(k3.SPEC, field)
    assert k31.SPEC.requires.server_args["moe_a2a_backend"] == (
        k3.SPEC.requires.server_args["moe_a2a_backend"]
    )
