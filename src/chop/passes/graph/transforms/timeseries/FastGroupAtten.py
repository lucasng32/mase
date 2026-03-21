"""
MASE transform pass: replace ``GroupSelfAttention`` with
``FastGroupSelfAttention``.

``group_ids`` must be provided in ``pass_args``.  The pass computes the group
partition once on CPU, writes it into each node's MASE metadata, and bakes it
into the replacement module so ``forward()`` never inspects the mask at
runtime.

Usage::

    from chop.passes.graph.transforms.timeseries import (
        fast_group_attention_transform_pass,
    )

    mg, info = fast_group_attention_transform_pass(
        mg, pass_args={"group_ids": torch.arange(64)}
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch

from chop.models.chronos2.layers import GroupSelfAttention
from chop.models.chronos2.optimized_layers import (
    FastGroupSelfAttention,
    GroupPartition,
    KernelDispatcher,
    KernelVariant,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Model analyser
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class GroupAttentionInfo:
    """Analysis result for a single graph node."""

    node_name: str
    module_path: str
    can_optimise: bool
    reason: str


class GroupAttentionAnalyser:
    """Walk a ``MaseGraph`` and identify nodes eligible for the grouped
    sparse attention transform."""

    @staticmethod
    def analyse(mg) -> list[GroupAttentionInfo]:
        """Return analysis info for every ``call_module`` node.

        Eligible nodes are those whose target module is an instance of
        ``GroupSelfAttention``.
        """
        results: list[GroupAttentionInfo] = []
        for node in mg.fx_graph.nodes:
            if node.op != "call_module":
                continue
            module = mg.model.get_submodule(node.target)
            if isinstance(module, GroupSelfAttention):
                results.append(
                    GroupAttentionInfo(
                        node_name=node.name,
                        module_path=node.target,
                        can_optimise=True,
                        reason="GroupSelfAttention detected",
                    )
                )
            elif isinstance(module, FastGroupSelfAttention):
                results.append(
                    GroupAttentionInfo(
                        node_name=node.name,
                        module_path=node.target,
                        can_optimise=False,
                        reason="Already optimised",
                    )
                )
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# Pass entry point
# ═══════════════════════════════════════════════════════════════════════════════
def fast_group_attention_transform_pass(
    mg,
    pass_args: dict | None = None,
) -> tuple:
    """Replace every ``GroupSelfAttention`` in *mg* with
    ``FastGroupSelfAttention``.

    Args:
        mg: ``MaseGraph`` wrapping a Chronos2Model.
        pass_args:
            group_ids (torch.Tensor, required):
                1-D long tensor of shape ``(B,)`` matching the batch size that
                will be used at inference.
            kernel_variant (str, optional):
                Force a specific kernel: ``"univariate"``, ``"triton"``, or
                ``"packed_sparse"``. Defaults to auto-selection via
                ``KernelDispatcher``.

    Returns:
        ``(mg, info_dict)`` where *info_dict* contains ``replaced`` count,
        the ``partition``, and per-node ``analysis``.
    """
    if pass_args is None:
        pass_args = {}

    group_ids: torch.Tensor | None = pass_args.get("group_ids")
    if group_ids is None:
        raise ValueError(
            "fast_group_attention_transform_pass requires 'group_ids' in "
            "pass_args.  Example: pass_args={'group_ids': torch.arange(64)}"
        )

    partition = GroupPartition.from_group_ids(group_ids)

    # Optional forced variant
    forced_variant_name: str | None = pass_args.get("kernel_variant")
    forced_variant: KernelVariant | None = None
    if forced_variant_name is not None:
        forced_variant = KernelVariant[forced_variant_name.upper()]

    analysis = GroupAttentionAnalyser.analyse(mg)

    replaced = 0
    for info in analysis:
        if not info.can_optimise:
            logger.debug("Skipping %s: %s", info.node_name, info.reason)
            continue

        node_target = info.module_path
        module = mg.model.get_submodule(node_target)

        # Determine device of the module being replaced
        device = next(module.parameters()).device

        variant = forced_variant or KernelDispatcher.select(partition, device)
        logger.info(
            "Replacing %s with FastGroupSelfAttention (variant=%s)",
            node_target,
            variant.name,
        )

        fast_module = FastGroupSelfAttention(
            config=module.self_attention.config,
            partition=partition,
            variant=variant,
        )
        fast_module.load_state_dict(module.state_dict(), strict=False)
        fast_module.to(device)

        # Write metadata for downstream passes
        node = _find_node_by_target(mg, node_target)
        if node is not None and "mase" in node.meta:
            ts_meta = node.meta["mase"].parameters.setdefault("timeseries", {})
            ts_meta["group_ids"] = group_ids.cpu()
            ts_meta["partition"] = partition
            ts_meta["kernel_variant"] = variant.name

        # Swap module in the model
        parent_path, _, attr = node_target.rpartition(".")
        parent = (
            mg.model.get_submodule(parent_path) if parent_path else mg.model
        )
        setattr(parent, attr, fast_module)
        replaced += 1

    mg.model.recompile()
    return mg, {
        "replaced": replaced,
        "partition": partition,
        "analysis": analysis,
    }


def _find_node_by_target(mg, target: str):
    for node in mg.fx_graph.nodes:
        if node.op == "call_module" and node.target == target:
            return node
    return None
