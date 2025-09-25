import pytest
import torch

from GenerativeBrainModel.models.gbm import GBM


def _build_model(d_model=32, d_stimuli=4, n_heads=4, n_layers=2, device="cpu", dtype=torch.float32):
    model = GBM(d_model=d_model, d_stimuli=d_stimuli, n_heads=n_heads, n_layers=n_layers)
    return model.to(device=device, dtype=dtype)


def _fake_batch(batch_size=2, seq_len=5, n_neurons=7, d_stimuli=4, device="cpu", dtype=torch.float32):
    torch.manual_seed(42)
    spikes = torch.rand(batch_size, seq_len, n_neurons, device=device, dtype=torch.float32) + 0.05
    positions = torch.randn(batch_size, n_neurons, 3, device=device, dtype=dtype)
    stim = torch.randn(batch_size, seq_len, d_stimuli, device=device, dtype=dtype)
    mask = torch.ones(batch_size, n_neurons, device=device, dtype=dtype)
    return spikes.to(dtype=dtype), stim, positions, mask


@pytest.fixture(scope="module")
def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("GBM attention stack relies on FlashAttention; CUDA required for test")
    return "cuda"


@pytest.mark.parametrize("dtype", [torch.float32])
def test_forward_returns_log_params(dtype, require_cuda):
    device = require_cuda
    model = _build_model(dtype=dtype, device=device)
    model.eval()
    spikes, stim, positions, mask = _fake_batch(dtype=dtype, device=device)

    mu, log_sigma, eta, log_delta = model(spikes, stim, positions, mask, get_logits=True)
    assert mu.shape == (spikes.shape[0], spikes.shape[1], spikes.shape[2])
    assert log_sigma.shape == mu.shape
    assert eta.shape == mu.shape
    assert log_delta.shape == mu.shape

    median_rates = model(spikes, stim, positions, mask, get_logits=False)
    assert median_rates.shape == mu.shape
    assert torch.all(median_rates > 0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_forward_handles_dtype(dtype, require_cuda):
    device = require_cuda
    model = _build_model(device=device, dtype=dtype)
    model.eval()
    spikes, stim, positions, mask = _fake_batch(device=device, dtype=torch.float32)

    spikes = spikes.to(device=device, dtype=dtype)
    stim = stim.to(device=device, dtype=dtype)
    positions = positions.to(device=device, dtype=dtype)
    mask = mask.to(device=device, dtype=dtype)

    outputs = model(spikes, stim, positions, mask, get_logits=True)
    for tensor in outputs:
        assert tensor.dtype == dtype


def test_autoregress_extends_sequence(require_cuda):
    device = require_cuda
    model = _build_model(device=device)
    model.eval()
    spikes, stim, positions, mask = _fake_batch(seq_len=6, device=device)

    n_steps = 3
    context_len = 4
    future = stim[:, context_len:context_len + n_steps, :]
    if future.shape[1] < n_steps:
        pad = torch.zeros(future.shape[0], n_steps - future.shape[1], future.shape[2], device=device, dtype=stim.dtype)
        future = torch.cat([future, pad], dim=1)

    generated = model.autoregress(
        init_x=spikes[:, :context_len, :],
        init_stimuli=stim[:, :context_len, :],
        point_positions=positions,
        neuron_pad_mask=mask,
        future_stimuli=future,
        n_steps=n_steps,
        context_len=context_len,
    )

    assert generated.shape[1] == context_len + n_steps
    assert torch.allclose(generated[:, :context_len, :], spikes[:, :context_len, :])


def test_position_normalization_unit_rms():
    model = _build_model()
    spikes, stim, positions, mask = _fake_batch()

    with torch.no_grad():
        centroid = positions.mean(dim=1, keepdim=True)
        pos_centered = positions - centroid
        r2 = (pos_centered ** 2).sum(dim=2)
        scale = (r2.mean(dim=1, keepdim=True).clamp_min(1e-6).sqrt()).unsqueeze(-1)
        rel = pos_centered / scale
    assert torch.allclose(rel.mean(dim=1), torch.zeros_like(rel.mean(dim=1)), atol=1e-4)
    rms = torch.sqrt(((rel ** 2).sum(dim=2).mean(dim=1)))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)

