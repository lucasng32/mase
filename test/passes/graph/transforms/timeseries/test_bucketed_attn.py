"""
Correctness tests for the TRITON_BUCKETED kernel variant.

Compares ``GroupAwareMHA`` with ``KernelVariant.TRITON_BUCKETED`` against:
  - the unmodified ``GroupSelfAttention`` baseline, and
  - the ``PACKED_SPARSE`` path (same dtype, tighter tolerance)

for a range of group configurations covering every bucket threshold.

All tests require CUDA.  Run with::

    pytest test/passes/graph/transforms/timeseries/test_bucketed_attn.py -v
"""

import pytest
import torch

from chop.models.chronos2.configuration_chronos2 import Chronos2CoreConfig
from chop.models.chronos2.layers import GroupSelfAttention
from chop.models.chronos2.optimized_layers import (
    GroupAwareMHA,
    GroupPartition,
    KernelVariant,
)
from chop.models.chronos2.triton_bucketed_attn import (
    BucketData,
    BucketedPartition,
    _bucket_for_size,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="TRITON_BUCKETED requires CUDA"
)

DEVICE = torch.device("cuda")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def cfg() -> Chronos2CoreConfig:
    return Chronos2CoreConfig(
        d_model=64,
        d_kv=16,
        d_ff=128,
        num_layers=1,
        num_heads=4,
        dropout_rate=0.0,
        attn_implementation="eager",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_mask(group_ids: torch.Tensor, T: int, dtype: torch.dtype) -> torch.Tensor:
    B = group_ids.shape[0]
    group_mask = (group_ids[:, None] == group_ids[None, :]).float()
    time_mask = torch.ones(B, T, device=group_ids.device)
    gtm = torch.einsum("qb,bt->qbt", group_mask, time_mask)
    gtm = gtm.permute(2, 0, 1).unsqueeze(1)
    return ((1.0 - gtm) * torch.finfo(dtype).min).to(dtype)


def _make_bucketed_mha(
    cfg: Chronos2CoreConfig,
    baseline_mha,
    group_ids: torch.Tensor,
) -> GroupAwareMHA:
    partition = GroupPartition.from_group_ids(group_ids)
    mha = GroupAwareMHA(cfg, partition, variant=KernelVariant.TRITON_BUCKETED).to(DEVICE)
    mha.load_state_dict(baseline_mha.state_dict())
    return mha


def _compare(
    cfg: Chronos2CoreConfig,
    group_ids: torch.Tensor,
    T: int = 8,
    atol: float = 2e-3,
):
    """Assert bucketed output matches baseline GroupSelfAttention."""
    group_ids = group_ids.to(DEVICE)
    baseline = GroupSelfAttention(cfg).to(DEVICE).eval()

    # Build optimized module with bucketed inner MHA
    partition = GroupPartition.from_group_ids(group_ids)
    optimized = GroupSelfAttention(cfg).to(DEVICE).eval()
    optimized.load_state_dict(baseline.state_dict())
    bucketed_mha = GroupAwareMHA(
        cfg, partition, variant=KernelVariant.TRITON_BUCKETED
    ).to(DEVICE)
    bucketed_mha.load_state_dict(baseline.self_attention.state_dict())
    optimized.self_attention = bucketed_mha

    B = group_ids.shape[0]
    torch.manual_seed(0)
    hs = torch.randn(B, T, cfg.d_model, device=DEVICE)
    mask = _make_mask(group_ids, T, hs.dtype)

    with torch.no_grad():
        ref = baseline(hs, mask).hidden_states
        out = optimized(hs, mask).hidden_states

    assert ref.shape == out.shape
    max_diff = (ref - out).abs().max().item()
    assert torch.allclose(ref, out, atol=atol), (
        f"max_diff={max_diff:.5f} (atol={atol}) for groups="
        f"{[g.tolist() for g in partition.groups]}"
    )


# ---------------------------------------------------------------------------
# BucketedPartition unit tests
# ---------------------------------------------------------------------------
class TestBucketForSize:
    def test_size_1_goes_to_bucket_2(self):
        assert _bucket_for_size(1) == 2

    def test_size_2_goes_to_bucket_2(self):
        assert _bucket_for_size(2) == 2

    def test_size_3_goes_to_bucket_4(self):
        assert _bucket_for_size(3) == 4

    def test_size_4_goes_to_bucket_4(self):
        assert _bucket_for_size(4) == 4

    def test_size_5_goes_to_bucket_8(self):
        assert _bucket_for_size(5) == 8

    def test_size_16_goes_to_bucket_16(self):
        assert _bucket_for_size(16) == 16

    def test_size_17_goes_to_bucket_32(self):
        assert _bucket_for_size(17) == 32

    def test_size_33_goes_to_bucket_64(self):
        assert _bucket_for_size(33) == 64

    def test_size_65_goes_to_bucket_128(self):
        assert _bucket_for_size(65) == 128


class TestBucketedPartitionConstruction:
    def test_uniform_pairs(self):
        gids = torch.tensor([0, 0, 1, 1, 2, 2])
        bp = BucketedPartition.from_group_ids(gids)
        assert len(bp.buckets) == 1
        assert 2 in bp.buckets
        assert bp.buckets[2].num_groups == 3
        assert bp.global_sort_perm.shape == (6,)
        assert bp.global_unsort_perm.shape == (6,)

    def test_unsort_is_inverse(self):
        gids = torch.tensor([0, 0, 1, 1, 1, 2])
        bp = BucketedPartition.from_group_ids(gids)
        identity = bp.global_unsort_perm[bp.global_sort_perm]
        assert torch.equal(identity, torch.arange(6))

    def test_mixed_sizes_produce_two_buckets(self):
        # sizes: 2, 4
        gids = torch.tensor([0, 0, 1, 1, 1, 1])
        bp = BucketedPartition.from_group_ids(gids)
        assert len(bp.buckets) == 2
        assert 2 in bp.buckets and 4 in bp.buckets

    def test_cu_seqlens_correctness(self):
        # 3 pairs → bucket 2, cu_seqlens = [0, 2, 4, 6]
        gids = torch.tensor([0, 0, 1, 1, 2, 2])
        bp = BucketedPartition.from_group_ids(gids)
        b2 = bp.buckets[2]
        assert b2.cu_seqlens.tolist() == [0, 2, 4, 6]
        assert b2.batch_offset == 0

    def test_batch_offsets_are_contiguous(self):
        # sizes: 2, 4 → bucket 2 has 2 elements, then bucket 4 has 4
        gids = torch.tensor([0, 0, 1, 1, 1, 1])
        bp = BucketedPartition.from_group_ids(gids)
        b2 = bp.buckets[2]
        b4 = bp.buckets[4]
        assert b2.batch_offset == 0
        assert b4.batch_offset == 2  # bucket 2 has 2 actual elements

    def test_univariate_in_bucket_2(self):
        gids = torch.arange(4)  # all size-1 groups → bucket 2
        bp = BucketedPartition.from_group_ids(gids)
        assert 2 in bp.buckets
        assert bp.buckets[2].num_groups == 4


# ---------------------------------------------------------------------------
# Correctness: TRITON_BUCKETED vs baseline GroupSelfAttention
# ---------------------------------------------------------------------------
class TestBucketedCorrectness:
    def test_all_pairs(self, cfg):
        _compare(cfg, torch.arange(8) // 2)

    def test_all_quads(self, cfg):
        _compare(cfg, torch.arange(8) // 4)

    def test_all_octets(self, cfg):
        _compare(cfg, torch.arange(16) // 8)

    def test_single_group_of_16(self, cfg):
        _compare(cfg, torch.zeros(16, dtype=torch.long))

    def test_single_univariate(self, cfg):
        _compare(cfg, torch.arange(4))

    def test_mixed_pairs_and_quads(self, cfg):
        # sizes: 2, 2, 4  → bucket 2 + bucket 4
        _compare(cfg, torch.tensor([0, 0, 1, 1, 2, 2, 2, 2]))

    def test_mixed_pairs_quads_octets(self, cfg):
        # sizes: 2, 4, 8
        gids = torch.tensor([0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2])
        _compare(cfg, gids)

    def test_mixed_with_univariate(self, cfg):
        # sizes: 1, 1, 2, 4
        gids = torch.tensor([0, 1, 2, 2, 3, 3, 3, 3])
        _compare(cfg, gids)

    def test_three_buckets(self, cfg):
        # sizes: 2, 4, 8
        gids = torch.tensor(
            [0, 0] + [1, 1, 1, 1] + [2, 2, 2, 2, 2, 2, 2, 2]
        )
        _compare(cfg, gids)

    def test_larger_batch_longer_seq(self, cfg):
        gids = torch.arange(32) // 4
        _compare(cfg, gids, T=16)

    def test_realistic_mixed_batch(self, cfg):
        # 10 pairs + 5 quads + 2 octets
        gids = torch.tensor(
            [i for i in range(10) for _ in range(2)]
            + [i for i in range(10, 15) for _ in range(4)]
            + [i for i in range(15, 17) for _ in range(8)]
        )
        _compare(cfg, gids, atol=3e-3)


# ---------------------------------------------------------------------------
# Bucketed agrees with packed-sparse (tighter tolerance, same dtype)
# ---------------------------------------------------------------------------
class TestBucketedVsPackedSparse:
    def _compare_variants(
        self,
        cfg: Chronos2CoreConfig,
        group_ids: torch.Tensor,
        T: int = 8,
        atol: float = 1e-3,
    ):
        group_ids = group_ids.to(DEVICE)
        partition = GroupPartition.from_group_ids(group_ids)

        base_mha = GroupAwareMHA(cfg, partition, variant=KernelVariant.PACKED_SPARSE).to(DEVICE)

        bucketed_mha = GroupAwareMHA(
            cfg, partition, variant=KernelVariant.TRITON_BUCKETED
        ).to(DEVICE)
        bucketed_mha.load_state_dict(base_mha.state_dict())

        B = group_ids.shape[0]
        torch.manual_seed(7)
        hs = torch.randn(B, T, cfg.d_model, device=DEVICE)
        mask_unused = torch.zeros(1)  # MHA forward doesn't use mask

        with torch.no_grad():
            ref = base_mha(hs, mask_unused).hidden_states
            out = bucketed_mha(hs, mask_unused).hidden_states

        max_diff = (ref - out).abs().max().item()
        assert torch.allclose(ref, out, atol=atol), (
            f"packed_sparse vs bucketed max_diff={max_diff:.5f}"
        )

    def test_pairs(self, cfg):
        self._compare_variants(cfg, torch.arange(8) // 2)

    def test_quads(self, cfg):
        self._compare_variants(cfg, torch.arange(8) // 4)

    def test_mixed_pairs_quads(self, cfg):
        self._compare_variants(cfg, torch.tensor([0, 0, 1, 1, 2, 2, 2, 2]))


# ---------------------------------------------------------------------------
# State dict / weight transfer
# ---------------------------------------------------------------------------
class TestBucketedStateDict:
    def test_load_from_mha_strict(self, cfg):
        """Weights load from plain MHA with strict=True."""
        from chop.models.chronos2.layers import MHA

        mha = MHA(cfg).to(DEVICE)
        partition = GroupPartition.from_group_ids(torch.arange(4) // 2)
        bucketed = GroupAwareMHA(cfg, partition, variant=KernelVariant.TRITON_BUCKETED).to(DEVICE)
        # Should not raise
        bucketed.load_state_dict(mha.state_dict())

    def test_index_buffers_not_in_state_dict(self, cfg):
        partition = GroupPartition.from_group_ids(torch.tensor([0, 0, 1, 1]))
        mha = GroupAwareMHA(cfg, partition, variant=KernelVariant.TRITON_BUCKETED).to(DEVICE)
        keys = set(mha.state_dict().keys())
        assert "_bucketed_sort_perm" not in keys
        assert "_bucketed_unsort_perm" not in keys


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
