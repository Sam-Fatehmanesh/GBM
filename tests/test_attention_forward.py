import sys
import os
import types
import importlib.util
import torch
import pytest


CUDA_AVAILABLE = torch.cuda.is_available()
BF16_SUPPORTED = CUDA_AVAILABLE and torch.cuda.get_device_capability()[0] >= 8

# Require CUDA and BF16 support for this test module
pytestmark = [
    pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required for these tests"),
    pytest.mark.skipif(not BF16_SUPPORTED, reason="BF16-capable GPU (SM80+) required"),
]


def install_flash_attn_stub():
    # Create stub package and interface module
    # If real flash_attn is importable, do nothing (we want real GPU kernel)
    try:
        import importlib

        importlib.import_module("flash_attn.flash_attn_interface")
        return
    except Exception:
        pass

    pkg = types.ModuleType("flash_attn")
    iface = types.ModuleType("flash_attn.flash_attn_interface")

    def flash_attn_varlen_kvpacked_func(
        q_packed,
        kv_packed,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        p_drop,
        softmax_scale=None,
        causal=False,
    ):
        # Return zeros with expected shape: (total_q, 1, Dh)
        return torch.zeros(
            (q_packed.shape[0], 1, q_packed.shape[-1]),
            device=q_packed.device,
            dtype=q_packed.dtype,
        )

    def flash_attn_varlen_qkvpacked_func(
        qkv, cu_seqlens, max_seqlen, dropout_p=0.0, softmax_scale=None, causal=False
    ):
        # Return zeros with expected shape: (Tdup, H, Dh)
        return torch.zeros(
            (qkv.shape[0], qkv.shape[2], qkv.shape[3]),
            device=qkv.device,
            dtype=qkv.dtype,
        )

    iface.flash_attn_varlen_kvpacked_func = flash_attn_varlen_kvpacked_func
    iface.flash_attn_varlen_qkvpacked_func = flash_attn_varlen_qkvpacked_func

    # Register in sys.modules before importing the target module
    sys.modules["flash_attn"] = pkg
    sys.modules["flash_attn.flash_attn_interface"] = iface


def _preload_minimal_gbm_modules():
    # Create lightweight parent packages to avoid executing package __init__ files
    if "GenerativeBrainModel" not in sys.modules:
        gbm_pkg = types.ModuleType("GenerativeBrainModel")
        gbm_pkg.__path__ = []  # mark as package-like
        sys.modules["GenerativeBrainModel"] = gbm_pkg
    if "GenerativeBrainModel.models" not in sys.modules:
        models_pkg = types.ModuleType("GenerativeBrainModel.models")
        models_pkg.__path__ = []
        sys.modules["GenerativeBrainModel.models"] = models_pkg

    # Preload rms module directly from file so attention can import RMSNorm without
    # triggering GenerativeBrainModel/models/__init__.py side effects.
    rms_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "GenerativeBrainModel", "models", "rms.py"
        )
    )
    if "GenerativeBrainModel.models.rms" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "GenerativeBrainModel.models.rms", rms_path
        )
        rms_mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(rms_mod)
        sys.modules["GenerativeBrainModel.models.rms"] = rms_mod


def import_attention_module():
    # Only stub if needed (on systems without flash_attn). On CUDA-enabled envs
    # with flash-attn installed, this no-ops due to the guard above.
    install_flash_attn_stub()
    _preload_minimal_gbm_modules()
    # Load the attention module directly from its file path to avoid
    # executing package __init__ side-effects that import unavailable symbols.
    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "GenerativeBrainModel",
            "models",
            "attention.py",
        )
    )
    spec = importlib.util.spec_from_file_location("attention_module", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_forward_early_return_no_spikes():
    attention = import_attention_module()
    SpikeSparseConnectomeRoutingAttention = (
        attention.SpikeSparseConnectomeRoutingAttention
    )

    d_model, n_heads = 32, 4
    model = SpikeSparseConnectomeRoutingAttention(
        d_model=d_model,
        n_heads=n_heads,
        neuron_cluster_size=2,
        num_clusters_per_head=2,
    )
    device = torch.device("cuda")
    model = model.to(device)
    model.eval()

    B, T, N = 2, 3, 5
    x = torch.randn(B, T, N, d_model, device=device)
    # Random positions (non-zero); the module normalizes internally
    positions = torch.randn(B, N, 3, device=device)
    neuron_pad_mask = torch.ones(B, N, dtype=torch.bool, device=device)
    spike_mask = torch.zeros(B, T, N, dtype=torch.bool, device=device)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y = model(x, positions, neuron_pad_mask, spike_mask)
    assert y.shape == x.shape
    # Early return path returns the original input tensor
    assert torch.allclose(y.float(), x.float())


def test_forward_with_spikes_runs_and_shapes_ok():
    attention = import_attention_module()
    SpikeSparseConnectomeRoutingAttention = (
        attention.SpikeSparseConnectomeRoutingAttention
    )

    d_model, n_heads = 32, 4
    model = SpikeSparseConnectomeRoutingAttention(
        d_model=d_model,
        n_heads=n_heads,
        neuron_cluster_size=2,
        num_clusters_per_head=2,
    )
    device = torch.device("cuda")
    model = model.to(device)
    model.eval()

    B, T, N = 2, 3, 5
    x = torch.randn(B, T, N, d_model, device=device)
    positions = torch.randn(B, N, 3, device=device)
    neuron_pad_mask = torch.ones(B, N, dtype=torch.bool, device=device)
    spike_mask = torch.zeros(B, T, N, dtype=torch.bool, device=device)
    spike_mask[0, 0, 0] = True  # ensure at least one spiking neuron

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y = model(x, positions, neuron_pad_mask, spike_mask)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_train_forward_and_backward_runs():
    attention = import_attention_module()
    SpikeSparseConnectomeRoutingAttention = (
        attention.SpikeSparseConnectomeRoutingAttention
    )

    d_model, n_heads = 32, 4
    model = SpikeSparseConnectomeRoutingAttention(
        d_model=d_model,
        n_heads=n_heads,
        neuron_cluster_size=2,
        num_clusters_per_head=2,
    )
    device = torch.device("cuda")
    model = model.to(device)
    model.train()

    B, T, N = 2, 3, 5
    x = torch.randn(B, T, N, d_model, device=device, requires_grad=True)
    positions = torch.randn(B, N, 3, device=device)
    neuron_pad_mask = torch.ones(B, N, dtype=torch.bool, device=device)
    spike_mask = torch.zeros(B, T, N, dtype=torch.bool, device=device)
    spike_mask[0, 0, 0] = True

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y = model(x, positions, neuron_pad_mask, spike_mask)
    loss = y.pow(2).mean()
    loss.backward()
    # Ensure gradients flowed to input and at least one parameter
    assert x.grad is not None and torch.isfinite(x.grad).all()
    any_param_has_grad = any((p.grad is not None) for p in model.parameters())
    assert any_param_has_grad


def test_torch_compile_eager_backend_forward():
    attention = import_attention_module()
    SpikeSparseConnectomeRoutingAttention = (
        attention.SpikeSparseConnectomeRoutingAttention
    )

    # Use a conservative compile config to maximize portability in CI
    import torch as _torch

    if not hasattr(_torch, "compile"):
        return  # skip if compile is unavailable

    d_model, n_heads = 32, 4
    model = SpikeSparseConnectomeRoutingAttention(
        d_model=d_model,
        n_heads=n_heads,
        neuron_cluster_size=2,
        num_clusters_per_head=2,
    )
    device = torch.device("cuda")
    model = model.to(device)
    model.eval()

    compiled = _torch.compile(model, backend="eager", mode="reduce-overhead")

    B, T, N = 2, 3, 5
    x = torch.randn(B, T, N, d_model, device=device)
    positions = torch.randn(B, N, 3, device=device)
    neuron_pad_mask = torch.ones(B, N, dtype=torch.bool, device=device)
    spike_mask = torch.zeros(B, T, N, dtype=torch.bool, device=device)
    spike_mask[0, 0, 0] = True

    # Run multiple forwards after compilation to ensure stability/caching
    for _ in range(8):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            y = compiled(x, positions, neuron_pad_mask, spike_mask)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()


def test_chunking_effective_batch_gt_65536():
    attention = import_attention_module()
    SpikeSparseConnectomeRoutingAttention = (
        attention.SpikeSparseConnectomeRoutingAttention
    )

    # Minimal dims to keep memory modest while forcing B_eff > 65536
    d_model, n_heads = 32, 1
    # Some GPUs/kernels can fail exactly at 65536; force chunk size below by tweaking constant
    if hasattr(attention, "MAX_BATCH_SIZE"):
        attention.MAX_BATCH_SIZE = 65535
    model = SpikeSparseConnectomeRoutingAttention(
        d_model=d_model,
        n_heads=n_heads,
        neuron_cluster_size=1,
        num_clusters_per_head=1,
    )
    device = torch.device("cuda")
    model = model.to(device)
    model.eval()

    # S = B*T; with H=1, B_eff = S*H = B. Choose B = 65537 to exceed chunk threshold.
    B, T, N = 65537, 1, 2
    x = torch.randn(B, T, N, d_model, device=device)
    positions = torch.randn(B, N, 3, device=device)  # non-zero to avoid padding
    neuron_pad_mask = torch.ones(B, N, dtype=torch.bool, device=device)
    spike_mask = torch.ones(
        B, T, N, dtype=torch.bool, device=device
    )  # ensure clusters are kept

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y = model(x, positions, neuron_pad_mask, spike_mask)
    assert y.shape == x.shape
    # With real FlashAttention, output should differ from input; just check finite
    assert torch.isfinite(y).all()
    assert not torch.allclose(y.float(), x.float())


def test_ssfa_huge_sparse_forward():
    attention = import_attention_module()
    SparseSpikeFullAttention = attention.SparseSpikeFullAttention

    # Runtime guard
    try:
        import flash_attn.flash_attn_interface as _fa

        _ = _fa.flash_attn_varlen_kvpacked_func
    except Exception:
        pytest.skip("flash-attn not available")

    device = torch.device("cuda")

    # Model (keep dims small to control memory)
    d_model, n_heads = 128, 4
    model = SparseSpikeFullAttention(d_model=d_model, n_heads=n_heads).to(device)
    model.eval()

    # Huge shapes per request
    B, T, N = 1, 192, 80000
    # Pad mask: keep only a small fraction to bound Q size
    keep_prob = 1.0
    spike_prob = 0.05

    # Memory guard: skip if little free VRAM
    try:
        free_bytes, _ = torch.cuda.mem_get_info(device)
        if free_bytes < 6 * (1024**3):
            pytest.skip("Insufficient free VRAM for huge SSFA test")
    except Exception:
        pass

    # Inputs
    x = torch.randn(B, T, N, d_model, device=device)
    positions = torch.randn(B, N, 3, device=device)
    neuron_pad_mask = torch.rand(B, N, device=device) < keep_prob
    spike_mask = torch.rand(B, T, N, device=device) < spike_prob

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y = model(x, positions, neuron_pad_mask, spike_mask)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_ssfa_huge_sparse_train_backward_and_compile():
    attention = import_attention_module()
    SparseSpikeFullAttention = attention.SparseSpikeFullAttention

    try:
        import flash_attn.flash_attn_interface as _fa

        _ = _fa.flash_attn_varlen_kvpacked_func
    except Exception:
        pytest.skip("flash-attn not available")

    device = torch.device("cuda")

    d_model, n_heads = 128, 4
    model = SparseSpikeFullAttention(d_model=d_model, n_heads=n_heads).to(device)

    B, T, N = 2, 96, 80000
    keep_prob = 1.0
    spike_prob = 0.05

    # Memory guard for backward
    try:
        free_bytes, _ = torch.cuda.mem_get_info(device)
        if free_bytes < 10 * (1024**3):
            pytest.skip("Insufficient free VRAM for huge SSFA backward test")
    except Exception:
        pass

    # Train forward + backward
    model.train()
    x = torch.randn(B, T, N, d_model, device=device, requires_grad=True)
    positions = torch.randn(B, N, 3, device=device)
    neuron_pad_mask = torch.rand(B, N, device=device) < keep_prob
    spike_mask = torch.rand(B, T, N, device=device) < spike_prob

    # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    #     y = model(x, positions, neuron_pad_mask, spike_mask)
    #     loss = y.pow(2).mean()
    # loss.backward()
    # assert x.grad is not None and torch.isfinite(x.grad).all()

    # Compile + train forward/backward
    import torch as _torch

    if hasattr(_torch, "compile"):
        compiled = _torch.compile(model, backend="eager", mode="reduce-overhead")
        x2 = torch.randn(B, T, N, d_model, device=device, requires_grad=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            y2 = compiled(x2, positions, neuron_pad_mask, spike_mask)
            loss2 = y2.pow(2).mean()
        loss2.backward()
        assert x2.grad is not None and torch.isfinite(x2.grad).all()


def test_neuron_causal_forward_backward_and_compile():
    attention = import_attention_module()
    NeuronCausalAttention = attention.NeuronCausalAttention

    device = torch.device("cuda")
    d_model, n_heads = 64, 8
    model = NeuronCausalAttention(d_model=d_model, n_heads=n_heads).to(device)

    B, T, N = 2, 256, 256
    x = torch.randn(B, T, N, d_model, device=device, requires_grad=True)
    neuron_pad_mask = torch.ones(
        B, N, dtype=torch.bool, device=device
    )  # no padding neurons

    # forward + backward
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y = model(x, neuron_pad_mask)
        loss = y.pow(2).mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    # compile forward
    import torch as _torch

    if hasattr(_torch, "compile"):
        compiled = _torch.compile(model, backend="eager", mode="reduce-overhead")
        x2 = torch.randn(B, T, N, d_model, device=device, requires_grad=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            y2 = compiled(x2, neuron_pad_mask)
            loss2 = y2.pow(2).mean()
        loss2.backward()
        assert x2.grad is not None and torch.isfinite(x2.grad).all()


def test_neuron_causal_huge_sparse_rows():
    attention = import_attention_module()
    NeuronCausalAttention = attention.NeuronCausalAttention

    try:
        _ = torch.cuda.mem_get_info()
    except Exception:
        pytest.skip("CUDA required")

    device = torch.device("cuda")
    d_model, n_heads = 64, 8
    model = NeuronCausalAttention(d_model=d_model, n_heads=n_heads, dropout_p=0.0).to(
        device
    )
    model.eval()

    # Large rows (B*N) with moderate T
    B, T, N = (
        1,
        96,
        80000,
    )  # B*N=32768 rows; with H=8 → BH=262144, so internal chunking should trigger
    x = torch.randn(B, T, N, d_model, device=device, requires_grad=True)
    neuron_pad_mask = torch.ones(
        B, N, dtype=torch.bool, device=device
    )  # no padding neurons

    # # Forward and backward
    # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    #     y = model(x, neuron_pad_mask)
    #     loss = y.pow(2).mean()
    # loss.backward()
    # assert y.shape == x.shape
    # assert torch.isfinite(y).all()
    # assert x.grad is not None and torch.isfinite(x.grad).all()

    # Compiled forward and backward if available
    import torch as _torch

    # if hasattr(_torch, "compile"):
    compiled = _torch.compile(model, backend="eager", mode="reduce-overhead")
    x2 = torch.randn(B, T, N, d_model, device=device, requires_grad=True)
    neuron_pad_mask2 = torch.ones(B, N, dtype=torch.bool, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y2 = compiled(x2, neuron_pad_mask2)
        loss2 = y2.pow(2).mean()
    loss2.backward()
    assert y2.shape == x2.shape
    assert torch.isfinite(y2).all()
    assert x2.grad is not None and torch.isfinite(x2.grad).all()


def test_huge_forward_backward_compile_spike_sparse_attention():
    # Runtime guard: require CUDA + bf16 and flash-attn available
    attention = import_attention_module()
    try:
        import flash_attn.flash_attn_interface as _fa

        _ = _fa.flash_attn_varlen_kvpacked_func
    except Exception:
        pytest.skip("flash-attn not available")

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("BF16-capable GPU (SM80+) required")

    device = torch.device("cuda")

    # Target sizes per request
    B, T, N = 2, 90, 80000
    d_model, n_heads = 64, 4
    neuron_cluster_size = 256
    num_clusters_per_head = 256

    # Memory safety check (rough lower bound):
    # Dominant cost is in/out centroid scores: (S*H*N*C) bf16 for each of in_score and out_score
    # Also account for inputs x and some working buffers.
    S = B * T
    Dh = d_model // n_heads
    bytes_bf16 = 2
    approx_scores = 2 * S * n_heads * N * num_clusters_per_head * bytes_bf16  # in + out
    approx_x = B * T * N * d_model * bytes_bf16
    # Add 20% headroom
    required_bytes = int((approx_scores + approx_x) * 1.2)

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    except Exception:
        # Fallback: skip unless huge GPUs
        pytest.skip("Could not query CUDA memory; skipping huge test")

    # if free_bytes < required_bytes:
    #     pytest.skip(f"Insufficient free VRAM ({free_bytes/1e9:.1f}GB < {required_bytes/1e9:.1f}GB) for huge test")

    SpikeSparseConnectomeRoutingAttention = (
        attention.SpikeSparseConnectomeRoutingAttention
    )
    model = SpikeSparseConnectomeRoutingAttention(
        d_model=d_model,
        n_heads=n_heads,
        neuron_cluster_size=neuron_cluster_size,
        num_clusters_per_head=num_clusters_per_head,
    ).to(device)

    positions = torch.randn(B, N, 3, device=device)
    neuron_pad_mask = torch.ones(B, N, dtype=torch.bool, device=device)

    # Spike mask set randomly per
    spike_prob = 0.05
    spike_mask = (
        torch.rand(B, T, N, device=device) < spike_prob
    )  # randomly True with probability spike_prob
    spike_mask = spike_mask.bool()

    # Forward (eval)
    model.eval()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        x = torch.randn(B, T, N, d_model, device=device)
        y = model(x, positions, neuron_pad_mask, spike_mask)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()

    # Forward+backward (train)
    model.train()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        x = torch.randn(B, T, N, d_model, device=device, requires_grad=True)
        y = model(x, positions, neuron_pad_mask, spike_mask)
        loss = y.pow(2).mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    # Compile and run one forward and backward pass
    if hasattr(torch, "compile"):
        compiled = torch.compile(model, backend="eager", mode="reduce-overhead")
        # Forward (eval)
        compiled.eval()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            x = torch.randn(B, T, N, d_model, device=device)
            y = compiled(x, positions, neuron_pad_mask, spike_mask)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()
        # Forward+backward (train)
        compiled.train()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            x = torch.randn(B, T, N, d_model, device=device, requires_grad=True)
            y = compiled(x, positions, neuron_pad_mask, spike_mask)
            loss = y.pow(2).mean()
        loss.backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
