"""
Correctness and integration tests for grouped sparse attention.

Compares ``FastGroupSelfAttention`` against the baseline
``GroupSelfAttention`` for a range of group configurations to verify that
the optimised paths (univariate, packed-sparse, triton, cuda) produce
numerically equivalent results.
"""

import pytest
import torch
import torch.nn as nn

from chop.models.chronos2.configuration_chronos2 import Chronos2CoreConfig
from chop.models.chronos2.layers import GroupSelfAttention
from chop.models.chronos2.cuda_grouped_attn import is_cuda_ext_available
from chop.models.chronos2.optimized_layers import (
    FastGroupSelfAttention,
    GroupPartition,
    KernelDispatcher,
    KernelVariant,
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


def _run_comparison(
    config: Chronos2CoreConfig,
    group_ids: torch.Tensor,
    variant: KernelVariant,
    device: torch.device,
    B: int | None = None,
    T: int = 8,
    atol: float = 1e-4,
):
    """Compare FastGroupSelfAttention against baseline GroupSelfAttention."""
    if B is None:
        B = group_ids.shape[0]
    group_ids = group_ids.to(device)
    dtype = torch.float32

    baseline = GroupSelfAttention(config).to(device).eval()
    partition = GroupPartition.from_group_ids(group_ids)
    fast = FastGroupSelfAttention(config, partition, variant=variant).to(device).eval()
    fast.load_state_dict(baseline.state_dict(), strict=False)

    torch.manual_seed(42)
    hs = torch.randn(B, T, config.d_model, device=device, dtype=dtype)
    mask = _make_group_time_mask(group_ids, T, dtype).to(device)

    with torch.no_grad():
        ref_out = baseline(hs, mask).hidden_states
        fast_out = fast(hs, mask).hidden_states

    max_diff = (ref_out - fast_out).abs().max().item()
    assert ref_out.shape == fast_out.shape, (
        f"Shape mismatch: {ref_out.shape} vs {fast_out.shape}"
    )
    assert torch.allclose(ref_out, fast_out, atol=atol), (
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
    def test_univariate_always_selected(self):
        p = GroupPartition.from_group_ids(torch.arange(4))
        assert KernelDispatcher.select(p, torch.device("cpu")) == KernelVariant.UNIVARIATE

    def test_cpu_selects_packed_sparse(self):
        p = GroupPartition.from_group_ids(torch.tensor([0, 0, 1, 1]))
        assert KernelDispatcher.select(p, torch.device("cpu")) == KernelVariant.PACKED_SPARSE

    @pytest.mark.skipif(
        not torch.cuda.is_available() or not is_cuda_ext_available(),
        reason="CUDA extension not available",
    )
    def test_cuda_selected_when_ext_available(self):
        p = GroupPartition.from_group_ids(torch.tensor([0, 0, 1, 1]))
        variant = KernelDispatcher.select(p, torch.device("cuda"), use_cuda_ext=True)
        assert variant == KernelVariant.CUDA

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_skipped_when_disabled(self):
        p = GroupPartition.from_group_ids(torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]))
        variant = KernelDispatcher.select(p, torch.device("cuda"), use_cuda_ext=False)
        # Should fall through to Triton (if available) or packed_sparse
        assert variant in (KernelVariant.TRITON, KernelVariant.PACKED_SPARSE)


# ---------------------------------------------------------------------------
# Correctness: univariate (all groups size 1)
# ---------------------------------------------------------------------------
class TestUnivariateCorrectness:
    def test_small_batch(self, small_config, device):
        _run_comparison(
            small_config,
            group_ids=torch.arange(4),
            variant=KernelVariant.UNIVARIATE,
            device=device,
        )

    def test_medium_batch(self, small_config, device):
        _run_comparison(
            small_config,
            group_ids=torch.arange(16),
            variant=KernelVariant.UNIVARIATE,
            device=device,
        )


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
        gids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        _run_comparison(
            small_config,
            group_ids=gids,
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
# Correctness: AOT CUDA kernel (CUDA + extension required)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_cuda_ext_available(),
    reason="CUDA extension not available",
)
class TestCudaCorrectness:
    def test_uniform_pairs(self, small_config):
        _run_comparison(
            small_config,
            group_ids=torch.arange(8) // 2,
            variant=KernelVariant.CUDA,
            device=torch.device("cuda"),
            atol=1e-4,
        )

    def test_uniform_quads(self, small_config):
        _run_comparison(
            small_config,
            group_ids=torch.arange(8) // 4,
            variant=KernelVariant.CUDA,
            device=torch.device("cuda"),
            atol=1e-4,
        )

    def test_mixed_group_sizes(self, small_config):
        _run_comparison(
            small_config,
            group_ids=torch.tensor([0, 0, 0, 1, 2, 2]),
            variant=KernelVariant.CUDA,
            device=torch.device("cuda"),
            atol=1e-4,
        )

    def test_single_group(self, small_config):
        _run_comparison(
            small_config,
            group_ids=torch.zeros(6, dtype=torch.long),
            variant=KernelVariant.CUDA,
            device=torch.device("cuda"),
            atol=1e-4,
        )

    def test_larger_batch(self, small_config):
        _run_comparison(
            small_config,
            group_ids=torch.arange(32) // 4,
            variant=KernelVariant.CUDA,
            device=torch.device("cuda"),
            T=16,
            atol=1e-4,
        )

    def test_univariate_group_auto_overrides_cuda(self, small_config):
        """All-univariate input should pick UNIVARIATE even if CUDA variant is requested."""
        partition = GroupPartition.from_group_ids(torch.arange(4))
        fast = FastGroupSelfAttention(
            small_config, partition, variant=KernelVariant.CUDA
        ).to(torch.device("cuda")).eval()
        # __init__ should have overridden variant to UNIVARIATE
        assert fast._variant == KernelVariant.UNIVARIATE


# ---------------------------------------------------------------------------
# Output shape and dtype consistency
# ---------------------------------------------------------------------------
class TestOutputProperties:
    def test_output_shape_matches_input(self, small_config, device):
        B, T = 6, 8
        group_ids = torch.tensor([0, 0, 1, 1, 2, 2])
        partition = GroupPartition.from_group_ids(group_ids)

        fast = FastGroupSelfAttention(
            small_config, partition, variant=KernelVariant.PACKED_SPARSE
        ).to(device).eval()

        hs = torch.randn(B, T, small_config.d_model, device=device)
        mask = _make_group_time_mask(group_ids.to(device), T, hs.dtype)

        with torch.no_grad():
            out = fast(hs, mask).hidden_states

        assert out.shape == (B, T, small_config.d_model)
        assert out.dtype == hs.dtype

    def test_output_dtype_preserved(self, small_config, device):
        B, T = 4, 4
        group_ids = torch.arange(B)
        partition = GroupPartition.from_group_ids(group_ids)

        fast = FastGroupSelfAttention(
            small_config, partition, variant=KernelVariant.UNIVARIATE
        ).to(device).eval()

        hs = torch.randn(B, T, small_config.d_model, device=device)
        mask = _make_group_time_mask(group_ids.to(device), T, hs.dtype)

        with torch.no_grad():
            out = fast(hs, mask).hidden_states

        assert out.dtype == hs.dtype


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
