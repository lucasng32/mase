import logging

import torch.nn as nn
import torch.fx as fx
from tqdm.contrib.logging import tqdm_logging_redirect
from torch.fx.experimental.optimization import (
    matches_module_pattern,
    replace_node_module,
)

from chop.models.chronos2.layers import TimeSelfAttention, GroupSelfAttention, FusedTimeGroupAttention
from chop.models.chronos2.configuration_chronos2 import Chronos2CoreConfig

logger = logging.getLogger(__name__)

def fused_time_group_attention_pass(graph, pass_args={}):
    """Perform Time & Group Attention fusion on the given graph.

    :param graph: a MaseGraph
    :type graph: MaseGraph

    :param pass_args: additional pass arguments
    :type pass_args: dict, optional

    :return: return a tuple of a MaseGraph and an empty dict
    :rtype: tuple(MaseGraph, dict)
    """
    PATTERN = (TimeSelfAttention, GroupSelfAttention)

    modules = dict(graph.model.named_modules())
    logger.debug(f"Found {len(modules)} modules in the model")

    total = len(graph.fx_graph.nodes)
    with tqdm_logging_redirect(total=total, loggers=[logger]) as pbar:
        pbar.set_description(f"Looking for pattern TimeSelfAttention -> GroupSelfAttention")

        for node in graph.fx_graph.nodes:
            if matches_module_pattern(PATTERN, node, modules):
                if len(node.args[0].users) > 1:
                    logger.warning("TimeSelfAttention output used by other nodes. Skipped!")
                    continue
                
                # node is GroupSelfAttention, node.args[0] is TimeSelfAttention
                time_node = node.args[0]
                group_node = node
                
                time_mod = modules[time_node.target]
                group_mod = modules[group_node.target]
                
                # We need to create a FusedTimeGroupAttention module and copy the weights
                config = time_mod.self_attention.config
                fused_mod = FusedTimeGroupAttention(config)
                
                # Copy weights
                fused_mod.time_attention.load_state_dict(time_mod.self_attention.state_dict())
                fused_mod.time_layer_norm.load_state_dict(time_mod.layer_norm.state_dict())
                fused_mod.time_dropout = time_mod.dropout
                
                fused_mod.group_attention.load_state_dict(group_mod.self_attention.state_dict())
                fused_mod.group_layer_norm.load_state_dict(group_mod.layer_norm.state_dict())
                fused_mod.group_dropout = group_mod.dropout
                
                fused_mod.to(next(time_mod.parameters()).device)
                fused_mod.train(time_mod.training)
                
                # Insert the new fused module into the modules dict and the FX graph
                replace_node_module(time_node, modules, fused_mod)
                
                # Update the node inputs/outputs
                # The Fused module takes the inputs of the Time module, plus maybe others
                # In Chronos2EncoderBlock, GroupSelfAttention takes (hidden_states, attention_mask=group_time_mask)
                # But our fused module takes:
                # hidden_states, position_ids, attention_mask, group_time_mask
                
                # time_node.args = (hidden_states, )
                # time_node.kwargs = {'position_ids': ..., 'attention_mask': ...}
                # group_node.args = (time_node.output, )
                # group_node.kwargs = {'attention_mask': group_time_mask}
                
                new_kwargs = {}
                new_kwargs.update(time_node.kwargs)
                if 'attention_mask' in group_node.kwargs:
                    new_kwargs['group_time_mask'] = group_node.kwargs['attention_mask']
                
                time_node.kwargs = new_kwargs
                
                # The output of the group node is what we should replace with the output of time_node
                group_node.replace_all_uses_with(time_node)
                graph.fx_graph.erase_node(group_node)
                
                pbar.update(1)
            pbar.update(1)

        graph.model = fx.GraphModule(graph.model, graph.fx_graph)
        pbar.set_description("Done")

    return graph, {}
