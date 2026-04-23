import pytest
import torch

pytest.importorskip("triton")

from mamba_ssm.modules import mamba3 as mamba3_module


def _fake_siso_combined(*args, **kwargs):
    _fake_siso_combined.calls.append(kwargs.get("memory_efficient"))
    q = kwargs.get("Q") if "Q" in kwargs else args[0]
    v = kwargs.get("V") if "V" in kwargs else args[2]
    batch, seqlen = q.shape[:2]
    nheads, headdim_v = v.shape[2], v.shape[3]
    return torch.zeros(batch, seqlen, nheads, headdim_v, device=q.device, dtype=v.dtype)


@pytest.mark.parametrize(
    "constructor_kwargs, expected_flag",
    [
        ({}, True),
        ({"use_mem_eff_path": False}, False),
    ],
)
def test_mamba3_forwards_memory_efficient_flag(monkeypatch, constructor_kwargs, expected_flag):
    _fake_siso_combined.calls = []
    monkeypatch.setattr(mamba3_module, "mamba3_siso_combined", _fake_siso_combined)

    model = mamba3_module.Mamba3(
        d_model=64,
        d_state=8,
        expand=1,
        headdim=64,
        is_mimo=False,
        device="cpu",
        dtype=torch.float32,
        **constructor_kwargs,
    ).eval()

    x = torch.randn(2, 3, 64)
    with torch.no_grad():
        y = model(x)

    assert y.shape == x.shape
    assert _fake_siso_combined.calls == [expected_flag]
