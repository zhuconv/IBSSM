import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "source_code" / "ibm2" / "modeling_ibm2.py"


def load_repo_ibm2_module():
    spec = importlib.util.spec_from_file_location("repo_ibm2_modeling", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class IdentityNorm(nn.Module):
    def forward(self, hidden_states, gate):
        return hidden_states


class FakeBernoulliIB(nn.Module):
    def __init__(self, dt_value=0.25, a_value=-0.5, log_decay_value=-0.7):
        super().__init__()
        self.dt_value = dt_value
        self.a_value = a_value
        self.log_decay_value = log_decay_value
        self._log_decay = None

    def forward(self, dt, dt_bias, A):
        self._log_decay = torch.full_like(dt, self.log_decay_value)
        return torch.full_like(dt, self.dt_value), torch.full_like(dt, self.a_value)

    def get_log_decay(self):
        return self._log_decay


class FakeGammaIB(nn.Module):
    def __init__(self, dt_value=2.0):
        super().__init__()
        self.dt_value = dt_value

    def forward(self, dt, dt_bias, A):
        return torch.full_like(dt, self.dt_value), A


def build_mixer(module, ib_type):
    config = module.IBM2Config(
        ib_type=ib_type,
        hidden_size=4,
        expand=1,
        num_heads=2,
        head_dim=2,
        state_size=2,
        num_hidden_layers=1,
        conv_kernel=2,
        n_groups=1,
        chunk_size=2,
        max_seq_length=8,
        use_bias=False,
        use_conv_bias=False,
    )
    config.ib_layers = [0]
    mixer = module.IBM2Mixer(config, layer_idx=0)
    mixer.norm = IdentityNorm()
    mixer.out_proj = nn.Identity()
    return mixer


def fake_causal_conv1d_fn(x, weight, bias, activation):
    return x


def test_bernoulli_training_prefill_uses_manual_chunk_path(monkeypatch):
    module = load_repo_ibm2_module()
    mixer = build_mixer(module, ib_type="bernoulli")
    mixer.ib = FakeBernoulliIB()
    mixer.train()

    captured = {}

    def fail_split_combined(*args, **kwargs):
        raise AssertionError("fused split+scan kernel should be bypassed when Bernoulli IB is active in training")

    def fail_chunk_combined(*args, **kwargs):
        raise AssertionError("combined chunk scan should be bypassed when Bernoulli IB returns token-wise decay")

    def fake_chunk_state(B, x, dt, dA_cumsum, states_in_fp32=True):
        captured["dt"] = dt.detach().clone()
        captured["dA_cumsum"] = dA_cumsum.detach().clone()
        batch, _, nheads, headdim = x.shape
        nchunks = dt.shape[2]
        dstate = B.shape[-1]
        base = x.mean(dim=1, keepdim=True)[..., None]
        return base.expand(batch, nchunks, nheads, headdim, dstate).contiguous()

    def fake_state_passing(states, dA_chunk_cumsum, initial_states=None):
        captured["dA_chunk_cumsum"] = dA_chunk_cumsum.detach().clone()
        adjusted = states + dA_chunk_cumsum.permute(0, 2, 1).unsqueeze(-1)
        return adjusted, adjusted[:, -1]

    def fake_chunk_scan(B, C, x, dt, dA_cumsum, prev_states, D=None, z=None):
        batch, seqlen, _, _ = x.shape
        dA_seq = dA_cumsum.permute(0, 2, 3, 1).reshape(batch, -1, x.shape[2])[:, :seqlen]
        return (x + dA_seq.unsqueeze(-1).to(x.dtype)).contiguous()

    monkeypatch.setattr(module, "causal_conv1d_fn", fake_causal_conv1d_fn)
    monkeypatch.setattr(module, "mamba_split_conv1d_scan_combined", fail_split_combined)
    monkeypatch.setattr(module, "mamba_chunk_scan_combined", fail_chunk_combined)
    monkeypatch.setattr(module, "chunk_state", fake_chunk_state)
    monkeypatch.setattr(module, "state_passing", fake_state_passing)
    monkeypatch.setattr(module, "chunk_scan", fake_chunk_scan)
    monkeypatch.setattr(module, "is_chunk_scan_available", True)

    hidden_states = torch.randn(2, 3, 4, requires_grad=True)
    out = mixer.cuda_kernels_forward(hidden_states)
    assert out.shape == hidden_states.shape

    out.sum().backward()
    assert hidden_states.grad is not None

    expected_log_decay = torch.full((2, 3, 2), -0.7)
    expected_log_decay = F.pad(expected_log_decay, (0, 0, 0, 1))
    expected_dA_cumsum = expected_log_decay.reshape(2, -1, 2, 2).permute(0, 3, 1, 2).float().cumsum(dim=-1)

    assert torch.allclose(captured["dA_cumsum"], expected_dA_cumsum)
    assert torch.allclose(captured["dA_chunk_cumsum"], expected_dA_cumsum[:, :, :, -1])


def test_gamma_training_uses_chunk_combined_and_skips_fused_kernel(monkeypatch):
    module = load_repo_ibm2_module()
    mixer = build_mixer(module, ib_type="gamma")
    mixer.ib = FakeGammaIB(dt_value=2.0)
    mixer.train()

    captured = {}

    def fail_split_combined(*args, **kwargs):
        raise AssertionError("fused split+scan kernel should be bypassed when IB modifies dt during training")

    def fail_manual_chunk(*args, **kwargs):
        raise AssertionError("Gamma IB should keep using mamba_chunk_scan_combined, not the manual Bernoulli path")

    def fake_chunk_combined(x, dt, A, B, C, chunk_size, **kwargs):
        captured["dt"] = dt.detach().clone()
        captured["A"] = A.detach().clone()
        dstate = B.shape[-1]
        final_states = torch.zeros(x.shape[0], x.shape[2], x.shape[3], dstate, dtype=x.dtype)
        return x.contiguous(), final_states

    monkeypatch.setattr(module, "causal_conv1d_fn", fake_causal_conv1d_fn)
    monkeypatch.setattr(module, "mamba_split_conv1d_scan_combined", fail_split_combined)
    monkeypatch.setattr(module, "mamba_chunk_scan_combined", fake_chunk_combined)
    monkeypatch.setattr(module, "chunk_state", fail_manual_chunk)
    monkeypatch.setattr(module, "state_passing", fail_manual_chunk)
    monkeypatch.setattr(module, "chunk_scan", fail_manual_chunk)

    hidden_states = torch.randn(2, 3, 4)
    out = mixer.cuda_kernels_forward(hidden_states)
    assert out.shape == hidden_states.shape
    assert torch.allclose(captured["dt"], torch.full_like(captured["dt"], 2.0))
