import copy
import math
from collections import defaultdict

import torch

from chop.ir.graph import MaseGraph
from chop.passes.graph import PASSES
from chop.passes.graph.transforms.quantize.quantize import QUANTIZEABLE_OP

from .modeling_chronos2 import Chronos2Output


CHRONOS2_FX_INPUT_NAMES = ["context", "group_ids", "future_covariates", "num_output_patches"]


def force_eager_attention_for_fx(model):
    if hasattr(model, "config") and hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"
    return model


def build_chronos2_dummy_input(
    model,
    batch_size: int = 1,
    device: str | torch.device | None = None,
    context_length: int | None = None,
    prediction_length: int | None = None,
):
    if device is None:
        device = model.device if hasattr(model, "device") else "cpu"

    context_length = context_length or int(model.chronos_config.context_length)
    output_patch_size = int(model.chronos_config.output_patch_size)
    prediction_length = prediction_length or output_patch_size
    num_output_patches = max(1, math.ceil(prediction_length / output_patch_size))
    future_length = num_output_patches * output_patch_size

    return {
        "context": torch.randn((batch_size, context_length), device=device, dtype=torch.float32),
        "context_mask": torch.ones((batch_size, context_length), device=device, dtype=torch.bool),
        "group_ids": torch.zeros((batch_size,), device=device, dtype=torch.long),
        "future_covariates": torch.zeros((batch_size, future_length), device=device, dtype=torch.float32),
        "future_covariates_mask": torch.zeros((batch_size, future_length), device=device, dtype=torch.bool),
        "future_target": torch.randn((batch_size, prediction_length), device=device, dtype=torch.float32),
        "future_target_mask": torch.ones((batch_size, prediction_length), device=device, dtype=torch.bool),
        "num_output_patches": num_output_patches,
    }


def build_chronos2_mase_graph(model, dummy_in: dict | None = None, hf_input_names: list[str] | None = None):
    dummy_in = dummy_in or build_chronos2_dummy_input(model)
    hf_input_names = hf_input_names or CHRONOS2_FX_INPUT_NAMES
    dummy_in = {key: value for key, value in dummy_in.items() if key in hf_input_names}
    mg = MaseGraph(model, hf_input_names=hf_input_names)
    mg, _ = PASSES["init_metadata"](mg)
    mg, _ = PASSES["add_common_metadata"](mg, pass_args={"dummy_in": dummy_in})
    return mg


def chronos2_node_inventory(mase_graph: MaseGraph) -> dict[str, list[dict]]:
    inventory = defaultdict(list)
    call_modules = mase_graph.modules

    for node in mase_graph.nodes:
        mase_meta = node.meta.get("mase")
        common = getattr(mase_meta, "parameters", {}).get("common", {}) if mase_meta else {}
        mase_op = common.get("mase_op")
        entry = {
            "name": node.name,
            "op": node.op,
            "target": str(node.target),
            "mase_op": mase_op,
        }

        if node.op == "call_module":
            module = call_modules.get(node.target)
            entry["module_type"] = type(module).__name__ if module is not None else None
        else:
            entry["module_type"] = None

        if mase_op in QUANTIZEABLE_OP:
            inventory["quantizable_now"].append(entry)
        elif entry["module_type"] in {"TimeSelfAttention", "GroupSelfAttention", "MHA"} or mase_op == "scaled_dot_product_attention":
            inventory["custom_candidates"].append(entry)
        else:
            inventory["traceable_not_quantized"].append(entry)

    return dict(inventory)


def attach_chronos2_graph_module_metadata(graph_module, reference_model, wrap_forward: bool = True):
    for attr in ("config", "chronos_config", "num_quantiles"):
        if hasattr(reference_model, attr):
            setattr(graph_module, attr, getattr(reference_model, attr))

    if hasattr(reference_model, "device"):
        graph_module.device = reference_model.device

    if wrap_forward:
        original_forward = graph_module.forward

        def _wrapped_forward(*args, **kwargs):
            output = original_forward(*args, **kwargs)
            if isinstance(output, dict) and not isinstance(output, Chronos2Output):
                return Chronos2Output(**output)
            return output

        graph_module.forward = _wrapped_forward

    return graph_module


def make_integer_quant_config(
    data_in_width: int = 8,
    data_in_frac_width: int = 4,
    weight_width: int = 8,
    weight_frac_width: int = 4,
    bias_width: int = 8,
    bias_frac_width: int = 4,
) -> dict:
    return {
        "name": "integer",
        "data_in_width": data_in_width,
        "data_in_frac_width": data_in_frac_width,
        "weight_width": weight_width,
        "weight_frac_width": weight_frac_width,
        "bias_width": bias_width,
        "bias_frac_width": bias_frac_width,
    }


def build_spectral_quant_config(
    mase_graph: MaseGraph,
    frac_width_by_module: dict[str, dict[str, int]],
    default_config: dict | None = None,
) -> dict:
    config = {
        "by": "regex_name",
        "default": {"config": copy.deepcopy(default_config or make_integer_quant_config())},
    }

    module_to_node_names: dict[str, list[str]] = defaultdict(list)
    for node in mase_graph.nodes:
        if node.op == "call_module":
            module_to_node_names[str(node.target)].append(node.name)

    for module_name, frac_info in frac_width_by_module.items():
        node_names = module_to_node_names.get(module_name, [])
        if not node_names:
            continue
        node_config = copy.deepcopy(default_config or make_integer_quant_config())
        node_config["data_in_frac_width"] = int(frac_info["data_in_frac_width"])
        node_config["weight_frac_width"] = int(frac_info["weight_frac_width"])
        node_config["bias_frac_width"] = int(frac_info.get("bias_frac_width", node_config["bias_frac_width"]))
        for node_name in node_names:
            config[node_name] = {"config": copy.deepcopy(node_config)}

    return config
