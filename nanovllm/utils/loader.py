import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    # Detect weight prefix for multimodal models
    # e.g. "model.language_model.X" → strip "language_model." → "model.X"
    strip_prefix = ""
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for k in f.keys():
                if ".language_model." in k:
                    strip_prefix = "language_model."
                break
        break
    loaded, skipped, errors = 0, 0, []
    for file in sorted(glob(os.path.join(path, "*.safetensors"))):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # Strip multimodal prefix, skip vision/mtp weights
                if strip_prefix:
                    if strip_prefix in weight_name:
                        weight_name_mapped = weight_name.replace(strip_prefix, "")
                    elif weight_name.startswith("model.") or "mtp." in weight_name or "visual" in weight_name:
                        # Skip vision/mtp weights but keep top-level weights like lm_head
                        skipped += 1
                        continue
                    else:
                        weight_name_mapped = weight_name
                else:
                    weight_name_mapped = weight_name
                # Skip expert weights from packed_modules_mapping matching
                skip_packing = "mlp.experts" in weight_name_mapped
                matched = False
                try:
                    if not skip_packing:
                        for k in packed_modules_mapping:
                            if k in weight_name_mapped:
                                v, shard_id = packed_modules_mapping[k]
                                param_name = weight_name_mapped.replace(k, v)
                                param = model.get_parameter(param_name)
                                weight_loader = getattr(param, "weight_loader")
                                if isinstance(shard_id, tuple):
                                    full_weight = f.get_tensor(weight_name)
                                    param_path = param_name.rsplit('.', 1)
                                    parent_mod = model.get_submodule(param_path[0]) if len(param_path) > 1 else model
                                    output_sizes = getattr(parent_mod, 'output_sizes', None)
                                    if output_sizes:
                                        split_sizes = [output_sizes[s] for s in shard_id]
                                    else:
                                        split_sizes = [full_weight.size(0) // len(shard_id)] * len(shard_id)
                                    chunks = full_weight.split(split_sizes, dim=0)
                                    for sid, chunk in zip(shard_id, chunks):
                                        weight_loader(param, chunk, sid)
                                else:
                                    weight_loader(param, f.get_tensor(weight_name), shard_id)
                                matched = True
                                loaded += 1
                                break
                    if not matched:
                        param = model.get_parameter(weight_name_mapped)
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, f.get_tensor(weight_name))
                        loaded += 1
                except Exception as e:
                    errors.append(f"{weight_name_mapped}: {e}")
    if errors:
        print(f"Weight loading: {loaded} loaded, {skipped} skipped, {len(errors)} ERRORS:")
        for e in errors[:20]:
            print(f"  ERROR: {e}")
    else:
        print(f"Weight loading: {loaded} loaded, {skipped} skipped, 0 errors")
