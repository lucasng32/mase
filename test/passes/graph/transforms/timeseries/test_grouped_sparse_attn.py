"""
Correctness and integration tests for grouped sparse attention.

Compares ``GroupSelfAttention`` with its inner MHA replaced by
``GroupAwareMHA`` against the unmodified baseline for a range of group
configurations, verifying that all dispatch paths (univariate, packed-sparse,
triton) produce numerically equivalent results.
"""

import pytest
import torch

from chop.models.chronos2.configuration_chronos2 import Chronos2CoreConfig
from chop.models.chronos2.layers import GroupSelfAttention, MHA, TimeSelfAttention
from chop.models.chronos2.optimized_layers import (
    GroupAwareMHA,
    GroupPartition,
    KernelDispatcher,
    KernelVariant,
    UnivariateGroupAwareMHA,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def small_config() -> Chronos2CoreConfig:
    """A tiny Chronos2 config for fast tests."""
    return Chronos2CoreConfig(
        d_model=64,
        d_kv=16,
        d_ff=128,
        num_layers=1,
        num_heads=4,
        dropout_rate=0.0,
        attn_implementation="eager",
    )


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_group_time_mask(
    group_ids: torch.Tensor, T: int, dtype: torch.dtype
) -> torch.Tensor:
    """Reproduce Chronos2Encoder._construct_and_invert_group_time_mask."""
    B = group_ids.shape[0]
    group_mask = (group_ids[:, None] == group_ids[None, :]).float()
    time_mask = torch.ones(B, T, device=group_ids.device)
    gtm = torch.einsum("qb,bt->qbt", group_mask, time_mask)
    gtm = gtm.permute(2, 0, 1).unsqueeze(1)
    gtm = (1.0 - gtm) * torch.finfo(dtype).min
    return gtm.to(dtype)


def _make_optimized(
    config: Chronos2CoreConfig,
    baseline: GroupSelfAttention,
    partition: GroupPartition,
    variant: KernelVariant,
    device: torch.device,
) -> GroupSelfAttention:
    """Return a GroupSelfAttention with its inner MHA replaced by GroupAwareMHA.

    Copies all weights from ``baseline`` so the comparison is fair.
    """
    optimized = GroupSelfAttention(config).to(device).eval()
    optimized.load_state_dict(baseline.state_dict())

    group_aware = GroupAwareMHA(config, partition, variant=variant).to(device)
    group_aware.load_state_dict(baseline.self_attention.state_dict())
    optimized.self_attention = group_aware

    return optimized


def _run_comparison(
    config: Chronos2CoreConfig,
    group_ids: torch.Tensor,
    variant: KernelVariant,
    device: torch.device,
    B: int | None = None,
    T: int = 8,
    atol: float = 1e-4,
):
    """Compare optimized GroupSelfAttention against the unmodified baseline."""
    if B is None:
        B = group_ids.shape[0]
    group_ids = group_ids.to(device)

    baseline = GroupSelfAttention(config).to(device).eval()
    partition = GroupPartition.from_group_ids(group_ids)
    optimized = _make_optimized(config, baseline, partition, variant, device)

    torch.manual_seed(42)
    hs = torch.randn(B, T, config.d_model, device=device, dtype=torch.float32)
    mask = _make_group_time_mask(group_ids, T, torch.float32).to(device)

    with torch.no_grad():
        ref_out = baseline(hs, mask).hidden_states
        opt_out = optimized(hs, mask).hidden_states

    max_diff = (ref_out - opt_out).abs().max().item()
    assert ref_out.shape == opt_out.shape, (
        f"Shape mismatch: {ref_out.shape} vs {opt_out.shape}"
    )
    assert torch.allclose(ref_out, opt_out, atol=atol), (
        f"Outputs differ by {max_diff:.6f} (atol={atol}) "
        f"for variant={variant.name}, groups={[g.tolist() for g in partition.groups]}"
    )


# ---------------------------------------------------------------------------
# GroupPartition tests
# ---------------------------------------------------------------------------
class TestGroupPartition:
    def test_univariate(self):
        gids = torch.arange(8)
        p = GroupPartition.from_group_ids(gids)
        assert p.num_groups == 8
        assert p.max_group_size == 1
        assert p.all_univariate is True

    def test_pairs(self):
        gids = torch.tensor([0, 0, 1, 1, 2, 2])
        p = GroupPartition.from_group_ids(gids)
        assert p.num_groups == 3
        assert p.max_group_size == 2
        assert p.all_univariate is False

    def test_mixed_sizes(self):
        gids = torch.tensor([0, 0, 0, 1, 2, 2])
        p = GroupPartition.from_group_ids(gids)
        assert p.num_groups == 3
        assert p.max_group_size == 3
        assert p.all_univariate is False

    def test_single_group(self):
        gids = torch.zeros(5, dtype=torch.long)
        p = GroupPartition.from_group_ids(gids)
        assert p.num_groups == 1
        assert p.max_group_size == 5
        assert p.all_univariate is False


# ---------------------------------------------------------------------------
# KernelDispatcher tests
# ---------------------------------------------------------------------------
class TestKernelDispatcher:
    def test_cpu_selects_packed_sparse(self):
        p = GroupPartition.from_group_ids(torch.tensor([0, 0, 1, 1]))
        assert KernelDispatcher.select(p, torch.device("cpu")) == KernelVariant.PACKED_SPARSE

    def test_univariate_on_cpu_selects_packed_sparse(self):
        # Univariate is handled by the pass, not the dispatcher.
        # On CPU the dispatcher falls through to PACKED_SPARSE regardless.
        p = GroupPartition.from_group_ids(torch.arange(4))
        assert KernelDispatcher.select(p, torch.device("cpu")) == KernelVariant.PACKED_SPARSE

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_selects_triton_bucketed(self):
        p = GroupPartition.from_group_ids(torch.tensor([0, 0, 1, 1]))
        variant = KernelDispatcher.select(p, torch.device("cuda"))
        assert variant in (KernelVariant.TRITON_BUCKETED, KernelVariant.PACKED_SPARSE)


# ---------------------------------------------------------------------------
# Correctness: univariate (all groups size 1) — via UnivariateGroupAwareMHA
# ---------------------------------------------------------------------------
class TestUnivariateCorrectness:
    def _run_univariate_comparison(self, config, group_ids, device, T=8, atol=1e-4):
        """Compare UnivariateGroupAwareMHA against the unmodified baseline."""
        group_ids = group_ids.to(device)
        B = group_ids.shape[0]

        baseline = GroupSelfAttention(config).to(device).eval()

        optimized = GroupSelfAttention(config).to(device).eval()
        optimized.load_state_dict(baseline.state_dict())
        mha_sd = baseline.self_attention.state_dict()
        univ_mha = UnivariateGroupAwareMHA(config).to(device)
        univ_mha.load_state_dict(
            {k: v for k, v in mha_sd.items() if k in ("v.weight", "o.weight")}
        )
        optimized.self_attention = univ_mha

        torch.manual_seed(42)
        hs = torch.randn(B, T, config.d_model, device=device)
        mask = _make_group_time_mask(group_ids, T, torch.float32).to(device)

        with torch.no_grad():
            ref = baseline(hs, mask).hidden_states
            out = optimized(hs, mask).hidden_states

        assert ref.shape == out.shape
        assert torch.allclose(ref, out, atol=atol), (
            f"max_diff={(ref-out).abs().max():.6f}"
        )

    def test_small_batch(self, small_config, device):
        self._run_univariate_comparison(small_config, torch.arange(4), device)

    def test_medium_batch(self, small_config, device):
        self._run_univariate_comparison(small_config, torch.arange(16), device)


# ---------------------------------------------------------------------------
# Correctness: packed sparse
# ---------------------------------------------------------------------------
class TestPackedSparseCorrectness:
    def test_uniform_pairs(self, small_config, device):
        _run_comparison(
            small_config,
            group_ids=torch.arange(8) // 2,
            variant=KernelVariant.PACKED_SPARSE,
            device=device,
        )

    def test_uniform_quads(self, small_config, device):
        _run_comparison(
            small_config,
            group_ids=torch.arange(8) // 4,
            variant=KernelVariant.PACKED_SPARSE,
            device=device,
        )

    def test_mixed_group_sizes(self, small_config, device):
        _run_comparison(
            small_config,
            group_ids=torch.tensor([0, 0, 0, 1, 2, 2]),
            variant=KernelVariant.PACKED_SPARSE,
            device=device,
        )

    def test_single_group(self, small_config, device):
        _run_comparison(
            small_config,
            group_ids=torch.zeros(6, dtype=torch.long),
            variant=KernelVariant.PACKED_SPARSE,
            device=device,
        )

    def test_many_small_groups(self, small_config, device):
        _run_comparison(
            small_config,
            group_ids=torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]),
            variant=KernelVariant.PACKED_SPARSE,
            device=device,
        )


# ---------------------------------------------------------------------------
# Correctness: Triton kernel (CUDA only)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton requires CUDA"
)
class TestTritonCorrectness:
    def test_uniform_pairs(self, small_config):
        _run_comparison(
            small_config,
            group_ids=torch.arange(8) // 2,
            variant=KernelVariant.TRITON,
            device=torch.device("cuda"),
            atol=1e-3,
        )

    def test_uniform_quads(self, small_config):
        _run_comparison(
            small_config,
            group_ids=torch.arange(8) // 4,
            variant=KernelVariant.TRITON,
            device=torch.device("cuda"),
            atol=1e-3,
        )

    def test_mixed_group_sizes(self, small_config):
        _run_comparison(
            small_config,
            group_ids=torch.tensor([0, 0, 0, 1, 2, 2]),
            variant=KernelVariant.TRITON,
            device=torch.device("cuda"),
            atol=1e-3,
        )

    def test_single_group(self, small_config):
        _run_comparison(
            small_config,
            group_ids=torch.zeros(6, dtype=torch.long),
            variant=KernelVariant.TRITON,
            device=torch.device("cuda"),
            atol=1e-3,
        )

    def test_larger_batch(self, small_config):
        _run_comparison(
            small_config,
            group_ids=torch.arange(32) // 4,
            variant=KernelVariant.TRITON,
            device=torch.device("cuda"),
            T=16,
            atol=1e-3,
        )


# ---------------------------------------------------------------------------
# Output shape and dtype consistency
# ---------------------------------------------------------------------------
class TestOutputProperties:
    def test_output_shape_matches_input(self, small_config, device):
        B, T = 6, 8
        group_ids = torch.tensor([0, 0, 1, 1, 2, 2]).to(device)
        partition = GroupPartition.from_group_ids(group_ids)

        baseline = GroupSelfAttention(small_config).to(device).eval()
        optimized = _make_optimized(
            small_config, baseline, partition, KernelVariant.PACKED_SPARSE, device
        )

        hs = torch.randn(B, T, small_config.d_model, device=device)
        mask = _make_group_time_mask(group_ids, T, hs.dtype)

        with torch.no_grad():
            out = optimized(hs, mask).hidden_states

        assert out.shape == (B, T, small_config.d_model)
        assert out.dtype == hs.dtype

    def test_output_dtype_preserved(self, small_config, device):
        B, T = 4, 4
        group_ids = torch.arange(B).to(device)

        baseline = GroupSelfAttention(small_config).to(device).eval()
        optimized = GroupSelfAttention(small_config).to(device).eval()
        optimized.load_state_dict(baseline.state_dict())
        mha_sd = baseline.self_attention.state_dict()
        univ_mha = UnivariateGroupAwareMHA(small_config).to(device)
        univ_mha.load_state_dict(
            {k: v for k, v in mha_sd.items() if k in ("v.weight", "o.weight")}
        )
        optimized.self_attention = univ_mha

        hs = torch.randn(B, T, small_config.d_model, device=device)
        mask = _make_group_time_mask(group_ids, T, hs.dtype)

        with torch.no_grad():
            out = optimized(hs, mask).hidden_states

        assert out.dtype == hs.dtype

    def test_univariate_handled_at_pass_not_runtime(self, small_config):
        """KernelDispatcher no longer has a UNIVARIATE variant — the pass
        creates UnivariateGroupAwareMHA directly."""
        assert not hasattr(KernelVariant, "UNIVARIATE")
        partition = GroupPartition.from_group_ids(torch.arange(4))
        # GroupAwareMHA with PACKED_SPARSE should NOT silently override to UNIVARIATE
        mha = GroupAwareMHA(small_config, partition, variant=KernelVariant.PACKED_SPARSE)
        assert mha._variant == KernelVariant.PACKED_SPARSE

    def test_swap_only_touches_group_self_attention(self, small_config):
        """Replacing GroupSelfAttention.self_attention must not affect
        the MHA inside TimeSelfAttention — they are distinct objects."""
        group_attn = GroupSelfAttention(small_config)
        time_attn = TimeSelfAttention(small_config)

        original_time_mha = time_attn.self_attention
        original_group_mha = group_attn.self_attention

        # Simulate what the MASE pass does: swap the inner MHA of GroupSelfAttention
        partition = GroupPartition.from_group_ids(torch.tensor([0, 0, 1, 1]))
        group_aware = GroupAwareMHA(small_config, partition, variant=KernelVariant.PACKED_SPARSE)
        group_aware.load_state_dict(group_attn.self_attention.state_dict())
        group_attn.self_attention = group_aware

        # GroupSelfAttention now has GroupAwareMHA
        assert isinstance(group_attn.self_attention, GroupAwareMHA)

        # TimeSelfAttention.self_attention is completely untouched
        assert time_attn.self_attention is original_time_mha
        assert isinstance(time_attn.self_attention, MHA)
        assert not isinstance(time_attn.self_attention, GroupAwareMHA)

        # The original GroupSelfAttention MHA is also a different object
        assert group_attn.self_attention is not original_group_mha


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
