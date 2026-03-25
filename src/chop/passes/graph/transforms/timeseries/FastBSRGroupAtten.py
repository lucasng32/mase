"""
MASE transform pass: swap the inner MHA of ``GroupSelfAttention`` with
``SparseGroupMHA``.

``group_ids`` must be provided in ``pass_args``. The pass computes the BSR 
group partition once, writes it into each node's MASE metadata, and replaces 
``group_self_attn.self_attention`` with a ``SparseGroupMHA`` instance so that
``forward()`` dynamically expands the metadata without inspecting the mask at runtime.
"""
import logging
from dataclasses import dataclass
import torch

from chop.models.chronos2.layers import GroupSelfAttention
from chop.models.chronos2.optimized_layers import SparseGroupMHA

logger = logging.getLogger(__name__)

@dataclass
class GroupAttentionInfo:
    node_name: str
    module_path: str
    can_optimise: bool
    reason: str

class GroupAttentionAnalyser:
    @staticmethod
    def analyse(mg) -> list[GroupAttentionInfo]:
        results: list[GroupAttentionInfo] = []
        for node in mg.fx_graph.nodes:
            if node.op != "call_module":
                continue
            module = mg.model.get_submodule(node.target)
            if not isinstance(module, GroupSelfAttention):
                continue
            if isinstance(module.self_attention, SparseGroupMHA):
                results.append(
                    GroupAttentionInfo(
                        node_name=node.name,
                        module_path=node.target,
                        can_optimise=False,
                        reason="Inner MHA already replaced with SparseGroupMHA",
                    )
                )
            else:
                results.append(
                    GroupAttentionInfo(
                        node_name=node.name,
                        module_path=node.target,
                        can_optimise=True,
                        reason="GroupSelfAttention with standard MHA detected",
                    )
                )
        return results

def fast_bsr_group_attention_transform_pass(mg, pass_args: dict | None = None) -> tuple:
    if pass_args is None:
        pass_args = {}

    group_ids: torch.Tensor | None = pass_args.get("group_ids")
    if group_ids is None:
        raise ValueError("Requires 'group_ids' in pass_args.")

    analysis = GroupAttentionAnalyser.analyse(mg)

    replaced = 0
    for info in analysis:
        if not info.can_optimise:
            continue

        module: GroupSelfAttention = mg.model.get_submodule(info.module_path)
        device = next(module.parameters()).device
        dtype = next(module.parameters()).dtype

        sparse_mha = SparseGroupMHA(
            config=module.self_attention.config,
            group_ids=group_ids
        )
        sparse_mha.load_state_dict(module.self_attention.state_dict())
        sparse_mha.to(device=device, dtype=dtype)

        module.self_attention = sparse_mha

        node = _find_node_by_target(mg, info.module_path)
        if node is not None:
            # Disconnect the dense mask from the graph so eliminate_dead_code can remove it
            if len(node.args) > 1:
                new_args = list(node.args)
                new_args[1] = None
                node.args = tuple(new_args)
            if 'attention_mask' in node.kwargs:
                new_kwargs = dict(node.kwargs)
                new_kwargs['attention_mask'] = None
                node.kwargs = new_kwargs
                
            if "mase" in node.meta:
                ts_meta = node.meta["mase"].parameters.setdefault("timeseries", {})
                ts_meta["group_ids"] = group_ids.cpu()
                ts_meta["bsr_fused"] = True

        replaced += 1

    mg.fx_graph.eliminate_dead_code()

    return mg, {
        "replaced": replaced,
        "analysis": analysis,
    }

def _find_node_by_target(mg, target: str):
    for node in mg.fx_graph.nodes:
        if node.op == "call_module" and node.target == target:
            return node
    return None
