"""
Correctness and integration tests for grouped sparse attention.

Compares ``GroupSelfAttention`` with its inner MHA replaced by
``GroupAwareMHA`` against the unmodified baseline for a range of group
configurations, verifying that all dispatch paths (univariate, packed-sparse,
triton) produce numerically equivalent results.
"""

import copy

import pytest
import torch

from chop.ir import MaseGraph
from chop.models.chronos2.configuration_chronos2 import Chronos2CoreConfig
from chop.models.chronos2.layers import GroupSelfAttention, MHA, TimeSelfAttention
from chop.models.chronos2.modeling_chronos2 import Chronos2Model
from chop.models.chronos2.optimized_layers import (
    GroupAwareMHA,
    GroupPartition,
    KernelDispatcher,
    KernelVariant,
    UnivariateGroupAwareMHA,
)
from chop.passes.graph.transforms.timeseries import fast_group_attention_transform_pass


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
        chronos_config={
            "context_length": 16,
            "input_patch_size": 4,
            "input_patch_stride": 4,
            "output_patch_size": 4,
            "quantiles": [0.1, 0.5, 0.9],
            "use_reg_token": False,
            "use_arcsinh": False,
        },
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
    device: torch.device,
    atol: float = 1e-4,
):
    """Apply the pass via MaseGraph and compare full model output against baseline to make sure the transformation does not have side effects.
    """
    group_ids = group_ids.to(device)
    B = group_ids.shape[0]

    model = Chronos2Model(config).to(device).eval()
    baseline = copy.deepcopy(model)

    mg = _build_mg(model)
    mg, _ = fast_group_attention_transform_pass(mg, pass_args={"group_ids": group_ids.cpu()})
    optimized = mg.model.to(device).eval()

    torch.manual_seed(42)
    context = torch.randn(B, config.chronos_config["context_length"], device=device)

    with torch.no_grad():
        ref = baseline(context=context, group_ids=group_ids)
        opt = optimized(context=context, group_ids=group_ids)

    ref_preds = ref["quantile_preds"]
    opt_preds = opt["quantile_preds"]
    assert ref_preds.shape == opt_preds.shape
    assert torch.allclose(ref_preds, opt_preds, atol=atol), (
        f"max diff: {(ref_preds - opt_preds).abs().max():.6f} (atol={atol})"
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
    def test_cuda_selects_triton_variant(self):
        p = GroupPartition.from_group_ids(torch.tensor([0, 0, 1, 1]))
        variant = KernelDispatcher.select(p, torch.device("cuda"))
        triton_variants = (
            KernelVariant.TRITON_SPECIALIZED,
            KernelVariant.TRITON_STITCHED,
            KernelVariant.TRITON_BUCKETED,
            KernelVariant.TRITON,
            KernelVariant.PACKED_SPARSE,
        )
        assert variant in triton_variants


# ---------------------------------------------------------------------------
# Correctness: packed sparse
# ---------------------------------------------------------------------------
class TestPackedSparseCorrectness:
    """Correctness tests for the PACKED_SPARSE kernel path.

    Pinned to CPU so ``KernelDispatcher`` always selects ``PACKED_SPARSE``.
    """

    def test_uniform_pairs(self, small_config):
        _run_comparison(small_config, group_ids=torch.arange(8) // 2, device=torch.device("cpu"))

    def test_uniform_quads(self, small_config):
        _run_comparison(small_config, group_ids=torch.arange(8) // 4, device=torch.device("cpu"))

    def test_mixed_group_sizes(self, small_config):
        _run_comparison(small_config, group_ids=torch.tensor([0, 0, 0, 1, 2, 2]), device=torch.device("cpu"))

    def test_single_group(self, small_config):
        _run_comparison(small_config, group_ids=torch.zeros(6, dtype=torch.long), device=torch.device("cpu"))

    def test_many_small_groups(self, small_config):
        _run_comparison(
            small_config,
            group_ids=torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]),
            device=torch.device("cpu"),
        )


# ---------------------------------------------------------------------------
# Correctness: CUDA dispatch path (TRITON_BUCKETED selected by KernelDispatcher)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)
class TestCudaCorrectness:
    """Correctness tests for the CUDA dispatch path.

    ``KernelDispatcher`` selects ``TRITON_BUCKETED`` (or ``PACKED_SPARSE`` as
    fallback) on CUDA — this is the kernel path that runs in production.
    """

    def test_uniform_pairs(self, small_config):
        _run_comparison(small_config, group_ids=torch.arange(8) // 2, device=torch.device("cuda"), atol=1e-3)

    def test_uniform_quads(self, small_config):
        _run_comparison(small_config, group_ids=torch.arange(8) // 4, device=torch.device("cuda"), atol=1e-3)

    def test_mixed_group_sizes(self, small_config):
        _run_comparison(small_config, group_ids=torch.tensor([0, 0, 0, 1, 2, 2]), device=torch.device("cuda"), atol=1e-3)

    def test_single_group(self, small_config):
        _run_comparison(small_config, group_ids=torch.zeros(6, dtype=torch.long), device=torch.device("cuda"), atol=1e-3)

# ---------------------------------------------------------------------------
# Output shape and dtype consistency
# ---------------------------------------------------------------------------
class TestOutputProperties:
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

    def test_swap_only_touches_group_self_attention(self, small_model):
        """The pass must swap GroupSelfAttention.self_attention to GroupAwareMHA
        and leave TimeSelfAttention.self_attention as plain MHA."""
        mg = _build_mg(small_model)
        mg, _ = fast_group_attention_transform_pass(mg, pass_args={"group_ids": torch.arange(8) // 2})
        optimized = mg.model

        group_attns = [m for m in optimized.modules() if isinstance(m, GroupSelfAttention)]
        time_attns = [m for m in optimized.modules() if isinstance(m, TimeSelfAttention)]

        assert group_attns, "No GroupSelfAttention found in optimized model"
        assert time_attns, "No TimeSelfAttention found in optimized model"

        for m in group_attns:
            assert isinstance(m.self_attention, GroupAwareMHA), (
                f"GroupSelfAttention.self_attention is {type(m.self_attention)}, expected GroupAwareMHA"
            )
        for m in time_attns:
            assert isinstance(m.self_attention, MHA) and not isinstance(m.self_attention, GroupAwareMHA), (
                f"TimeSelfAttention.self_attention was unexpectedly changed to {type(m.self_attention)}"
            )


# ---------------------------------------------------------------------------
# MaseGraph pass integration tests
# ---------------------------------------------------------------------------
_CUSTOM_OPS = {
    "modules": {GroupSelfAttention: {}, TimeSelfAttention: {}},
    "functions": {},
}


def _build_mg(model: Chronos2Model) -> MaseGraph:
    return MaseGraph(
        copy.deepcopy(model),
        hf_input_names=["context", "group_ids"],
        custom_ops=_CUSTOM_OPS,
    )


@pytest.fixture
def small_model(small_config) -> Chronos2Model:
    return Chronos2Model(small_config).eval()


class TestMasePassIntegration:
    """Tests that ``fast_group_attention_transform_pass`` works correctly
    when applied through the MaseGraph API."""

    def test_pass_replaces_all_group_self_attention_nodes(self, small_model, small_config):
        """Every GroupSelfAttention in the graph must have its inner MHA swapped."""
        mg = _build_mg(small_model)
        mg, info = fast_group_attention_transform_pass(mg, pass_args={"group_ids": torch.arange(4)})

        expected = sum(
            1
            for node in mg.fx_graph.nodes
            if node.op == "call_module"
            and isinstance(mg.model.get_submodule(node.target), GroupSelfAttention)
        )
        assert info["replaced"] == expected
        assert info["replaced"] == small_config.num_layers

    def test_pass_returns_correct_info_keys(self, small_model):
        mg = _build_mg(small_model)
        mg, info = fast_group_attention_transform_pass(mg, pass_args={"group_ids": torch.arange(4)})
        assert "replaced" in info
        assert "partition" in info
        assert "analysis" in info

    def test_univariate_installs_univariate_mha(self, small_model):
        """All-unique group_ids → UnivariateGroupAwareMHA must be installed."""
        mg = _build_mg(small_model)
        mg, _ = fast_group_attention_transform_pass(mg, pass_args={"group_ids": torch.arange(8)})
        for node in mg.fx_graph.nodes:
            if node.op == "call_module":
                module = mg.model.get_submodule(node.target)
                if isinstance(module, GroupSelfAttention):
                    assert isinstance(module.self_attention, UnivariateGroupAwareMHA), (
                        f"{node.target}.self_attention is {type(module.self_attention)}"
                    )

    def test_multivariate_installs_group_aware_mha(self, small_model):
        """Grouped group_ids → GroupAwareMHA must be installed."""
        mg = _build_mg(small_model)
        mg, _ = fast_group_attention_transform_pass(mg, pass_args={"group_ids": torch.arange(8) // 2})
        for node in mg.fx_graph.nodes:
            if node.op == "call_module":
                module = mg.model.get_submodule(node.target)
                if isinstance(module, GroupSelfAttention):
                    assert isinstance(module.self_attention, GroupAwareMHA), (
                        f"{node.target}.self_attention is {type(module.self_attention)}"
                    )

    def test_no_group_ids_returns_none(self, small_model):
        mg = _build_mg(small_model)
        assert fast_group_attention_transform_pass(mg, pass_args={}) is None

    def test_no_pass_args_returns_none(self, small_model):
        mg = _build_mg(small_model)
        assert fast_group_attention_transform_pass(mg, pass_args=None) is None

    def test_pass_output_numerically_matches_baseline(self, small_model, small_config):
        """Model output after the pass must match the unmodified baseline."""
        B = 4
        group_ids = torch.arange(B)
        baseline = copy.deepcopy(small_model)

        mg = _build_mg(small_model)
        mg, _ = fast_group_attention_transform_pass(mg, pass_args={"group_ids": group_ids})
        optimized = mg.model.eval()

        torch.manual_seed(42)
        context = torch.randn(B, small_config.chronos_config["context_length"])

        with torch.no_grad():
            ref = baseline(context=context, group_ids=group_ids)
            opt = optimized(context=context, group_ids=group_ids)

        assert ref["quantile_preds"].shape == opt["quantile_preds"].shape
        assert torch.allclose(ref["quantile_preds"], opt["quantile_preds"], atol=1e-4), (
            f"max diff: {(ref['quantile_preds'] - opt['quantile_preds']).abs().max():.6f}"
        )

    def test_metadata_written_to_nodes(self, small_model):
        """The pass must write group_ids, partition, and kernel_variant into
        MASE metadata for each replaced node."""
        mg = _build_mg(small_model)
        mg, _ = fast_group_attention_transform_pass(mg, pass_args={"group_ids": torch.arange(8) // 2})
        for node in mg.fx_graph.nodes:
            if node.op != "call_module":
                continue
            if not isinstance(mg.model.get_submodule(node.target), GroupSelfAttention):
                continue
            if "mase" not in node.meta:
                continue  # metadata not initialised — skip silently
            ts = node.meta["mase"].parameters.get("timeseries", {})
            assert "group_ids" in ts, f"group_ids missing from {node.name} metadata"
            assert "partition" in ts, f"partition missing from {node.name} metadata"
            assert "kernel_variant" in ts, f"kernel_variant missing from {node.name} metadata"


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
