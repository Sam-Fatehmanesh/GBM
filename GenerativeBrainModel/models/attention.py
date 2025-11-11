import torch
import torch.nn as nn
import torch.nn.functional as F
from GenerativeBrainModel.models.rms import RMSNorm
from torch.nn.attention import sdpa_kernel, SDPBackend
import math
from flash_attn.flash_attn_interface import (
    flash_attn_varlen_qkvpacked_func as flash_varlen_qkv,
)
from flash_attn.flash_attn_interface import flash_attn_varlen_kvpacked_func


MAX_BATCH_SIZE = 65535


# ------------------------- TorchDynamo compile helpers -------------------------
@torch._dynamo.disable
def _compute_max_lens_tensors(lens_q: torch.Tensor, lens_k: torch.Tensor):
    """
    Compute and return the maximum values of lens_q and lens_k as Python ints.
    """
    max_q = int(lens_q.max().item()) if lens_q.numel() > 0 else 0
    max_k = int(lens_k.max().item()) if lens_k.numel() > 0 else 0
    return max_q, max_k


@torch._dynamo.disable
def _build_varlen_metadata_from_masks(keep_bt: torch.Tensor, send_bt: torch.Tensor):
    """
    Construct index and sequence metadata from boolean masks for efficient sparse attention.

    Given two boolean masks of shape (S, N):
      - keep_bt: Mask indicating valid query tokens, True at (b, t) if this row should be attended to as a query.
      - send_bt: Mask indicating valid key/value tokens, True at (b, t) if this row can be attended to as key/value.

    Returns:
        idx_q: 1D indices of (b, t) pairs for valid queries, suitable for advanced indexing.
        idx_kv: 1D indices of (b, t) pairs for valid keys/values.
        cu_q: Cumulative sum offsets for queries, shape (S+1,)
        cu_k: Cumulative sum offsets for key/values, shape (S+1,)
        lens_q: Number of valid queries per batch element, shape (S,)
        lens_k: Number of valid keys per batch element, shape (S,)
        max_q: Maximum number of queries in any batch element (int)
        max_k: Maximum number of keys in any batch element (int)
    """
    S, N = keep_bt.shape
    device = keep_bt.device
    lens_q = keep_bt.sum(dim=1, dtype=torch.int32)
    lens_k = send_bt.sum(dim=1, dtype=torch.int32)
    cu_q = torch.zeros(S + 1, dtype=torch.int32, device=device)
    cu_k = torch.zeros(S + 1, dtype=torch.int32, device=device)
    if S > 0:
        cu_q[1:] = torch.cumsum(lens_q, dim=0)
        cu_k[1:] = torch.cumsum(lens_k, dim=0)
    max_q, max_k = _compute_max_lens_tensors(lens_q, lens_k)
    arange_SN = torch.arange(S * N, device=device, dtype=torch.long).reshape(S, N)
    idx_q = arange_SN[keep_bt].reshape(-1).contiguous()
    idx_kv = arange_SN[send_bt].reshape(-1).contiguous()
    return idx_q, idx_kv, cu_q, cu_k, lens_q, lens_k, max_q, max_k


@torch._dynamo.disable
def _index_copy_rows(
    out: torch.Tensor, out_valid: torch.Tensor, valid_rows_mask: torch.Tensor
):
    """Avoid boolean index_put in compiled region by using index_copy with int indices."""
    if valid_rows_mask.dtype is not torch.bool:
        valid_rows_mask = valid_rows_mask != 0
    if not bool(valid_rows_mask.any().item()):
        return out
    row_idx = torch.where(valid_rows_mask)[0].to(dtype=torch.long)
    return out.index_copy(0, row_idx, out_valid)


@torch._dynamo.disable
def _indices_from_mask(mask_1d: torch.Tensor):
    """Return 1D long indices of True positions from a boolean mask, out of graph."""
    if mask_1d.dtype is not torch.bool:
        mask_1d = mask_1d != 0
    return torch.where(mask_1d)[0].to(dtype=torch.long).contiguous()


class SparseSpikeFullAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_rope_features: int = 32,  # number of (direction, freq) angles available for RoPE
        dropout: float = 0.0,
        # ---- new, all optional (keeps Dh fixed) ----
        pos_tail_dim: int = 16,  # Dp: channels per head reserved for positional tail (<= Dh)
        n_rff: int = 32,  # M: random Fourier features; tail uses a learned (2M -> Dp) compression
        rff_sigma: float = 1.0,  # bandwidth for RFF wrt your RMS-scaled coords
        pos_tail_scale: float = 0.1,  # γ: strength of positional kernel in logits
        n_rot_pairs: int
        | None = None,  # m: rotary pairs per head (first 2m dims rotated)
    ):
        super().__init__()

        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = float(dropout)

        self.norm = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # ---- RoPE basis (post-projection; direction-sensitive) ----
        self.n_rope_features = int(n_rope_features)
        dirs = torch.randn(self.n_rope_features, 3)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        # tune upper band to your scale; 1..1e4 is fine for wide range, but you can lower upper band
        freqs = torch.logspace(
            math.log10(1.0), math.log10(10000.0), self.n_rope_features
        )
        self.register_buffer("rope_dirs", dirs, persistent=False)
        self.register_buffer("rope_freqs", freqs, persistent=False)

        # ---- In-place positional tail (relative bias inside dot product), keeps Dh fixed ----
        # Random Fourier features Ω ~ N(0, σ^2 I), φ(p) = [cos(Ωp), sin(Ωp)] ∈ R^{2M}
        self.n_rff = int(n_rff)
        Omega = torch.randn(self.n_rff, 3) * float(rff_sigma)
        self.register_buffer("rff_Omega", Omega, persistent=False)

        # compress 2M -> Dp once (shared across heads) + per-head gain
        self.pos_tail_dim = int(pos_tail_dim)  # Dp
        assert 0 <= self.pos_tail_dim <= self.head_dim, (
            "pos_tail_dim must be <= head_dim"
        )
        self.pos_C = nn.Linear(
            2 * self.n_rff, self.pos_tail_dim, bias=False
        )  # shared compression
        self.pos_head_gain = nn.Parameter(
            torch.ones(self.n_heads, self.pos_tail_dim)
        )  # per-head scaling
        self.pos_tail_scale = float(pos_tail_scale)  # γ

        # ---- Rotary slice config (first 2m dims rotated; tail occupies last Dp dims) ----
        # we must leave room: 2m + Dp <= Dh
        max_m = max(0, (self.head_dim - self.pos_tail_dim) // 2)
        self.n_rot_pairs = (
            int(n_rot_pairs) if n_rot_pairs is not None else min(16, max_m)
        )
        self.n_rot_pairs = min(self.n_rot_pairs, max_m)  # safety

    # -------- helpers --------
    @torch.no_grad()
    def _rope_angles(self, positions):
        rope_dirs = self.rope_dirs.to(dtype=positions.dtype, device=positions.device)
        rope_freqs = self.rope_freqs.to(
            dtype=positions.dtype, device=positions.device
        )  # (F,)
        proj = torch.einsum("bnd,fd->bnf", positions, rope_dirs)  # (B,N,F)
        angles = proj * rope_freqs  # broadcast (B,N,F)
        return angles  # (B,N,F)

    @torch.no_grad()
    def _rff_phi(self, positions):  # positions: (B, N, 3)
        # φ(p) = [cos(Ωp), sin(Ωp)] with Ω: (M,3)
        Omega = self.rff_Omega.to(
            dtype=positions.dtype, device=positions.device
        )  # (M,3)
        proj = torch.einsum("bnd,md->bnm", positions, Omega)  # (B,N,M)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)  # (B,N,2M)

    @staticmethod
    def _apply_rotary_inplace(x, theta, m):
        """
        Apply in-place rotary positional embedding to the first 2*m dimensions of each attention head.

        Args:
            x (Tensor): Input tensor of shape (total_tokens, n_heads, head_dim).
            theta (Tensor): Rotary angles of shape (total_tokens, F). Only the first m angles are used.
            m (int): Number of rotary pairs. The first 2*m channels per head are rotated.

        Returns:
            Tensor: The input tensor x with rotary embedding applied to first 2*m dims.
        """

        if m <= 0:
            return x

        x_even = x[..., : 2 * m : 2]  # (total, H, m)
        x_odd = x[..., 1 : 2 * m : 2]  # (total, H, m)
        sin = torch.sin(theta[:, :m]).unsqueeze(1)  # (total, 1, m)
        cos = torch.cos(theta[:, :m]).unsqueeze(1)  # (total, 1, m)

        x[..., : 2 * m : 2] = x_even * cos - x_odd * sin
        x[..., 1 : 2 * m : 2] = x_even * sin + x_odd * cos
        return x

    def forward(self, x, point_positions, neuron_pad_mask, spike_mask):
        """
        Forward pass for attention module with rotary and RFF-based spatial embeddings.

        Args:
            x (Tensor): Input activations of shape (B, T, N, D).
            point_positions (Tensor): Neuron positions, shape (B, N, 3).
            neuron_pad_mask (Tensor): Mask for valid (unpadded) neurons, shape (B, N), nonzero for valid.
            spike_mask (Tensor): Mask for "spiking" positions/tokens, shape (B, T, N), nonzero for spikes.

        Returns:
            Tensor:
                Output tensor after attention, preserving input shape and padding:
                of shape (B, T, N, D).

        Note:
            - Non-spiking, valid tokens are used as queries; spiking, valid as keys/values.
            - Rotary (RoPE) and Random Fourier Features (RFF) positional information are injected into Q/K.
            - In-place, varlen sparse attention is computed with memory and compute optimizations.
            - Handles variable batch sizes, variable numbers of neurons (N), and masking.
        """
        # x: (B,T,N,D), positions: (B,N,3), masks as described
        B, T, N, D = x.shape
        H, Dh = self.n_heads, self.head_dim
        S = B * T
        m = self.n_rot_pairs
        Dp = self.pos_tail_dim

        # 1) Normalize activations (not positions)
        xn = self.norm(x)

        # 2) Precompute RoPE angles and φ(p) once per (B,N)
        angles_BNF = self._rope_angles(point_positions)  # (B,N,F_rope)
        phi_BN2M = self._rff_phi(point_positions)  # (B,N,2M)

        # 3) Flatten to (S,N,*) for (B,T) "sequence"
        xn_bt = xn.view(B, T, N, D).reshape(S, N, D)

        # 4) Build masks and compact indices BEFORE projections
        #    Queries: NON-spiking, valid neurons only. KV: spiking, valid neurons only.
        valid_bt = (neuron_pad_mask != 0).unsqueeze(1).expand(B, T, N).reshape(S, N)
        spiking_bt = (spike_mask != 0).reshape(S, N) & valid_bt
        keep_bt = valid_bt  # & (~spiking_bt)   # remove spiking neurons from Q
        send_bt = spiking_bt
        idx_q, idx_kv, cu_q, cu_k, lens_q, lens_k, max_q, max_k = (
            _build_varlen_metadata_from_masks(keep_bt, send_bt)
        )

        # 5) Compact inputs for projections
        x_flat = xn_bt.reshape(S * N, D)
        pre_q = x_flat.index_select(0, idx_q)  # (total_q, D)
        pre_kv = x_flat.index_select(0, idx_kv)  # (total_k, D)

        # 6) Project to Q/K/V
        q = self.q_proj(pre_q).view(-1, H, Dh)  # (total_q,H,Dh)
        k = self.k_proj(pre_kv).view(-1, H, Dh)  # (total_k,H,Dh)
        v = self.v_proj(pre_kv).view(-1, H, Dh)  # (total_k,H,Dh)

        # 7) Gather angles for compacted tokens and apply RoPE to first 2m dims (post-proj)
        angles_SN = (
            angles_BNF.unsqueeze(1).expand(B, T, N, -1).reshape(S * N, -1)
        )  # (S*N, F_rope)
        theta_q = angles_SN.index_select(0, idx_q)  # (total_q, F_rope)
        theta_k = angles_SN.index_select(0, idx_kv)  # (total_k, F_rope)
        self._apply_rotary_inplace(q, theta_q, m)
        self._apply_rotary_inplace(k, theta_k, m)

        # 8) In-place positional tail (distance/sector kernel) in the LAST Dp dims of Q/K; zero same slice in V
        if Dp > 0:
            # compress φ(p) -> Dp using shared linear, then per-head gain; scale by sqrt(γ)
            phi_SN = (
                phi_BN2M.unsqueeze(1).expand(B, T, N, -1).reshape(S * N, -1)
            )  # (S*N, 2M)
            phi_q = phi_SN.index_select(0, idx_q)  # (total_q, 2M)
            phi_k = phi_SN.index_select(0, idx_kv)  # (total_k, 2M)

            tail_q = self.pos_C(phi_q)  # (total_q, Dp)
            tail_k = self.pos_C(phi_k)  # (total_k, Dp)

            # apply per-head gain and global scale √γ, then overwrite last Dp dims
            scale = self.pos_tail_scale**0.5
            # (total, H, Dp)
            q_tail = scale * (tail_q.unsqueeze(1) * self.pos_head_gain.unsqueeze(0))
            k_tail = scale * (tail_k.unsqueeze(1) * self.pos_head_gain.unsqueeze(0))

            q[..., Dh - Dp :] = q_tail
            k[..., Dh - Dp :] = k_tail
            v[..., Dh - Dp :] = 0.0  # ensure tail affects logits only

        # 9) FlashAttention-2 varlen (unchanged)
        q_packed = q.contiguous()  # (total_q,H,Dh)
        kv_packed = torch.stack([k, v], dim=1).contiguous()  # (total_k,2,H,Dh)
        p_drop = self.dropout if self.training else 0.0

        attn_out = flash_attn_varlen_kvpacked_func(
            q_packed,
            kv_packed,
            cu_q,
            cu_k,
            max_q,
            max_k,
            p_drop,
            softmax_scale=None,
            causal=False,
        )  # (total_q,H,Dh)

        # 10) (Optional) zero positional tail in out before o_proj (should already be ~0 since V tail is 0)
        if Dp > 0:
            zero_tail = (
                attn_out[..., :Dp].detach().new_zeros(attn_out.shape[:-1] + (Dp,))
            )
            attn_out = torch.cat([attn_out[..., : Dh - Dp], zero_tail], dim=-1)

        # 11) Scatter back to (S,N,D) and output proj + residual
        out_heads = torch.zeros(S * N, H, Dh, device=x.device, dtype=attn_out.dtype)
        out_heads = out_heads.index_copy(0, idx_q, attn_out)
        out_D = out_heads.view(S * N, D)
        out = self.o_proj(out_D).view(B, T, N, D)

        # Add residual connection
        return out + x


class NeuronCausalAttention(nn.Module):
    """
    Causal self-attention over time per neuron independently, with timewise RoPE.
    - Operates on each neuron timeline (length T) with causal attention.
    - Respects neuron_pad_mask (B,N): skip padded neurons entirely to save compute.
    - Spike mask is ignored.
    - Launches at most (2^18) sequences per FLASH call by chunking rows.
    - Adds rotary positional embeddings over time axis per neuron.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout_p: float = 0.0,
        max_bh_per_call: int = (2**16 - 1),
        max_seq_len: int = 512,
        # --- Gating controls (optional) ---
        gate_capacity_k: int | None = None,
        gate_temperature: float = 0.0,
        state_rank: int = 16,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = float(dropout_p)
        self.max_bh_per_call = int(max_bh_per_call)

        self.norm = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # --- Gating params ---
        self.gate_capacity_k = gate_capacity_k
        self.gate_temperature = gate_temperature
        self.state_rank = state_rank
        if self.gate_capacity_k is not None:
            # Learned gate vector over neuron embedding
            self.gate_vector = nn.Parameter(torch.zeros(d_model))
            # Low-rank state embedding over neuron axis (N,D) -> (D), created lazily on first forward
            self.state_neuron_proj1: nn.Linear | None = None  # (N -> R)
            self.state_neuron_proj2: nn.Linear | None = None  # (R -> 1)
            nn.init.normal_(self.gate_vector, mean=0.0, std=(1.0 / (d_model**0.5)))
            # Note: the neuron-axis projections will be initialized when created

        # --- Timewise RoPE (rotary) ---
        self.max_seq_len = max_seq_len
        # Precompute sin/cos tables for rotary, with 32 rotary dims per head by default (or as many as fit)
        self.rotary_frac = 0.5
        self.rotary_dim = int(self.head_dim * self.rotary_frac)
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim)
        )
        t = torch.arange(self.max_seq_len)
        freqs = torch.outer(t, inv_freq)  # (T, rotary_dim // 2)
        sin, cos = freqs.sin(), freqs.cos()
        self.register_buffer("rope_sin", sin, persistent=False)
        self.register_buffer("rope_cos", cos, persistent=False)

    def _to_bhld(self, x):  # (B*, L, D) -> (B*, H, L, Dh)
        Bx, L, D = x.shape
        return (
            x.view(Bx, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        )

    def _from_bhld(self, x):  # (B*, H, L, Dh) -> (B*, L, D)
        Bx, H, L, Dh = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(Bx, L, H * Dh)

    def _apply_timewise_rope(self, x, seq_len):
        """
        Apply rotary positional embeddings along the time axis.
        x: (batch*, H, seq_len, Dh)
        Only applies to the first rotary_dim of Dh.
        """
        rotary_dim = self.rotary_dim
        if rotary_dim == 0:
            return x
        # get sin/cos for this seq_len, shape (seq_len, rotary_dim // 2)
        sin = self.rope_sin[:seq_len].to(dtype=x.dtype, device=x.device)
        cos = self.rope_cos[:seq_len].to(dtype=x.dtype, device=x.device)
        # Shape to (1,1,seq_len, rotary_dim), broadcast along batch/H
        sin = sin[None, None, :, :]
        cos = cos[None, None, :, :]

        x1 = x[..., :rotary_dim]
        x2 = x[..., rotary_dim:]
        # Reshape for pairwise treatment
        x1 = x1.reshape(*x.shape[:-1], rotary_dim // 2, 2)
        sin, cos = sin, cos
        x1_0 = x1[..., 0]
        x1_1 = x1[..., 1]
        rope_x0 = x1_0 * cos - x1_1 * sin
        rope_x1 = x1_0 * sin + x1_1 * cos
        # Stack and reshape back
        x1_rope = torch.stack([rope_x0, rope_x1], dim=-1).reshape(
            *x.shape[:-1], rotary_dim
        )
        # Concatenate with remaining dims
        return torch.cat([x1_rope, x2], dim=-1)

    # @torch._dynamo.disable
    def forward(self, x: torch.Tensor, neuron_pad_mask: torch.Tensor):
        # x: (B, T, N, D)
        B, T, N, D = x.shape
        xn = self.norm(x)
        xt = xn.permute(0, 2, 1, 3).contiguous().view(B * N, T, D)  # (B*N, T, D)
        res = xt

        # Row filter from neuron_pad_mask
        valid_rows = neuron_pad_mask.reshape(-1) != 0

        out = torch.zeros_like(xt)
        if valid_rows.any():
            q_in = xt[valid_rows]
            k_in = q_in
            v_in = q_in

            # Projections
            q = self._to_bhld(self.q_proj(q_in))  # (Bv, H, L, Dh)
            k = self._to_bhld(self.k_proj(k_in))
            v = self._to_bhld(self.v_proj(v_in))

            # Apply timewise RoPE to q/k
            q = self._apply_timewise_rope(q, seq_len=q.shape[2])
            k = self._apply_timewise_rope(k, seq_len=k.shape[2])

            # Ensure FLASH dtype
            if q.dtype not in (torch.bfloat16, torch.float16):
                q = q.to(torch.bfloat16)
            if k.dtype not in (torch.bfloat16, torch.float16):
                k = k.to(torch.bfloat16)
            if v.dtype not in (torch.bfloat16, torch.float16):
                v = v.to(torch.bfloat16)

            Bv, H, L, Dh = q.shape
            if H > self.max_bh_per_call:
                raise RuntimeError(
                    f"n_heads={H} exceeds max_bh_per_call={self.max_bh_per_call}; reduce heads or increase cap"
                )

            # --- Optional neuron-time gating before attention ---
            use_gating = self.gate_capacity_k is not None and self.gate_capacity_k > 0
            if use_gating:
                # Compute per-(B,T,N) logits using gate vector and low-rank state embedding
                # xn: (B,T,N,D)
                x_for_gate = xn
                # Mask padded neurons to avoid leaking into state summary
                pad_mask_bt = neuron_pad_mask.unsqueeze(1).expand(B, T, N)  # (B,T,N)
                x_masked = x_for_gate.masked_fill(~pad_mask_bt.unsqueeze(-1), 0)

                # Build low-rank linear map over neuron axis: (N,D) -> (D)
                # Do this by operating independently per feature: reshape to (B*T*D, N)
                x_feat = x_masked.permute(0, 1, 3, 2).contiguous().view(B * T * D, N)
                # Lazily create projections bound to current N
                if (self.state_neuron_proj1 is None) or (
                    self.state_neuron_proj1.in_features != N
                ):
                    self.state_neuron_proj1 = nn.Linear(
                        N, self.state_rank, bias=False
                    ).to(x.device, dtype=x.dtype)
                    self.state_neuron_proj2 = nn.Linear(
                        self.state_rank, 1, bias=False
                    ).to(x.device, dtype=x.dtype)
                    nn.init.xavier_uniform_(self.state_neuron_proj1.weight)
                    nn.init.xavier_uniform_(self.state_neuron_proj2.weight)

                state_weights = self.state_neuron_proj2(
                    self.state_neuron_proj1(x_feat)
                )  # (B*T*D,1)
                state_embed = state_weights.view(B, T, D).unsqueeze(2)  # (B,T,1,D)
                gate_inp = x_for_gate + state_embed  # (B,T,N,D)
                logits = torch.einsum(
                    "btnd,d->btn", gate_inp, self.gate_vector
                )  # (B,T,N)
                # Mask invalid neurons
                logits = logits.masked_fill(~pad_mask_bt, float("-inf"))
                # Slight stochastic sampling via Gumbel noise during training
                if self.training and (self.gate_temperature > 0.0):
                    # Gumbel(0,1) noise
                    u = torch.rand_like(logits, dtype=torch.float32)
                    g = -torch.log(-torch.log(u.clamp_(min=1e-6, max=1 - 1e-6)))
                    logits = logits + self.gate_temperature * g.to(logits.dtype)
                k_cap = min(self.gate_capacity_k, N)
                # Top-k over neurons per (B,T)
                topk = torch.topk(logits, k=k_cap, dim=2, largest=True, sorted=False)
                selected = torch.zeros_like(logits, dtype=torch.bool)
                selected.scatter_(2, topk.indices, True)
                # Ensure we never select padded entries
                selected = selected & pad_mask_bt
                # Build (B*N, T) selection aligned with rows
                selected_bnT = selected.permute(0, 2, 1).contiguous().view(B * N, T)
                selected_bnT_valid = selected_bnT[valid_rows]  # (Bv, T)

                # Pack selected tokens across rows for varlen self-attn
                sel_positions = torch.nonzero(
                    selected_bnT_valid, as_tuple=False
                )  # (total, 2) [row, t]
                if sel_positions.numel() > 0:
                    row_idx = sel_positions[:, 0]
                    t_idx = sel_positions[:, 1]
                    # Gather selected Q/K/V tokens: (total,H,Dh)
                    qs_sel = q[row_idx, :, t_idx, :]
                    ks_sel = k[row_idx, :, t_idx, :]
                    vs_sel = v[row_idx, :, t_idx, :]
                    # Pack QKV for flash varlen
                    qkv_packed = torch.stack(
                        [qs_sel, ks_sel, vs_sel], dim=1
                    ).contiguous()  # (total,3,H,Dh)
                    # cu_seqlens from per-row counts
                    lens = selected_bnT_valid.sum(dim=1, dtype=torch.int32)  # (Bv,)
                    cu = torch.zeros(Bv + 1, dtype=torch.int32, device=x.device)
                    cu[1:] = torch.cumsum(lens, dim=0)
                    max_len = int(lens.max().item()) if lens.numel() > 0 else 0
                    p_drop = self.dropout_p if self.training else 0.0
                    # FlashAttention-2 varlen self-attn with causal constraint
                    attn_sel = flash_varlen_qkv(
                        qkv_packed,
                        cu,
                        max_len,
                        p_drop,
                        softmax_scale=None,
                        causal=True,
                    )  # (total,H,Dh)
                    # Scatter back into (Bv,H,L,Dh), leaving non-selected steps as zero (cheap path -> residual only)
                    out_valid_bhld = q.new_zeros((Bv, H, L, Dh))
                    out_valid_bhld[row_idx, :, t_idx, :] = attn_sel
                    out_valid = self._from_bhld(out_valid_bhld).to(res.dtype)
                else:
                    # No tokens selected anywhere: leave out_valid as zeros => identity path after residual
                    out_valid = torch.zeros_like(q_in)
            else:
                # No gating: full causal SDPA per row chunked by heads
                # Choose row chunk so (rows_chunk * H) <= max_bh_per_call
                rows_step = max(1, min(Bv, self.max_bh_per_call // H))
                out_chunks = []
                for i in range(0, Bv, rows_step):
                    qs = q[i : i + rows_step]
                    ks = k[i : i + rows_step]
                    vs = v[i : i + rows_step]
                    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                        out_chunk = F.scaled_dot_product_attention(
                            qs, ks, vs, is_causal=True, dropout_p=self.dropout_p
                        )
                    out_chunks.append(out_chunk)
                out_valid = torch.cat(out_chunks, dim=0)  # (Bv,H,L,Dh)
                out_valid = self._from_bhld(out_valid).to(res.dtype)

            out = _index_copy_rows(out, out_valid, valid_rows)

        out = out + res
        out = out.view(B, N, T, D).permute(0, 2, 1, 3).contiguous()
        return self.o_proj(out)
