import math
import os

import torch
from safetensors import safe_open


BYTES_PER_BLOCK = 16
FP4_VALUES = [
    +0.0,
    +0.5,
    +1.0,
    +1.5,
    +2.0,
    +3.0,
    +4.0,
    +6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def _parameter_name_map(num_layers: int) -> dict[str, str | tuple[str, str]]:
    mapping: dict[str, str | tuple[str, str]] = {}
    for layer_idx in range(num_layers):
        prefix = f"block.{layer_idx}.mlp"
        mapping[f"{prefix}.mlp1_bias"] = f"{prefix}.mlp1_bias"
        mapping[f"{prefix}.mlp1_weight"] = (
            f"{prefix}.mlp1_weight.blocks",
            f"{prefix}.mlp1_weight.scales",
        )
        mapping[f"{prefix}.mlp2_bias"] = f"{prefix}.mlp2_bias"
        mapping[f"{prefix}.mlp2_weight"] = (
            f"{prefix}.mlp2_weight.blocks",
            f"{prefix}.mlp2_weight.scales",
        )
    return mapping


class Checkpoint:
    def __init__(
        self, path: str, device: torch.device, num_layers: int
    ) -> None:
        device_str = (
            device.type
            if device.index is None
            else f"{device.type}:{device.index}"
        )
        self.device_str = device_str
        self.parameter_name_map = _parameter_name_map(num_layers)

        files = [
            os.path.join(path, filename)
            for filename in os.listdir(path)
            if filename.endswith(".safetensors")
        ]
        if not files:
            raise FileNotFoundError(f"No .safetensors files found in {path}")

        tensor_name_to_file: dict[str, str] = {}
        for filename in files:
            with safe_open(filename, framework="pt", device=device_str) as handle:
                for key in handle.keys():
                    tensor_name_to_file[key] = filename
        self.tensor_name_to_file = tensor_name_to_file

    def get(self, name: str) -> torch.Tensor:
        mapped_name = self.parameter_name_map.get(name, name)
        if isinstance(mapped_name, tuple):
            return self._get_mxfp4_tensor(*mapped_name, dtype=torch.bfloat16)
        return self._get_tensor(mapped_name)

    def _get_tensor(self, name: str) -> torch.Tensor:
        if name not in self.tensor_name_to_file:
            raise KeyError(f"Tensor {name} not found in checkpoint")
        with safe_open(
            self.tensor_name_to_file[name],
            framework="pt",
            device=self.device_str,
        ) as handle:
            return handle.get_tensor(name)

    def _get_mxfp4_tensor(
        self,
        blocks_name: str,
        scales_name: str,
        *,
        dtype: torch.dtype,
        rows_per_chunk: int = 16384 * 512,
    ) -> torch.Tensor:
        blocks = self._get_tensor(blocks_name)
        scales = self._get_tensor(scales_name).to(torch.int32) - 127
        if blocks.shape[:-1] != scales.shape:
            raise ValueError(
                f"{blocks.shape=} does not match {scales.shape=}"
            )

        lookup = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)
        *prefix_shape, groups, packed_width = blocks.shape
        rows_total = math.prod(prefix_shape) * groups
        blocks = blocks.reshape(rows_total, packed_width)
        scales = scales.reshape(rows_total, 1)
        output = torch.empty(
            rows_total,
            packed_width * 2,
            dtype=dtype,
            device=blocks.device,
        )

        for row_start in range(0, rows_total, rows_per_chunk):
            row_end = min(row_start + rows_per_chunk, rows_total)
            block_chunk = blocks[row_start:row_end]
            scale_chunk = scales[row_start:row_end]
            low = (block_chunk & 0x0F).to(torch.long)
            high = (block_chunk >> 4).to(torch.long)
            output_chunk = output[row_start:row_end]
            output_chunk[:, 0::2] = lookup[low]
            output_chunk[:, 1::2] = lookup[high]
            torch.ldexp(output_chunk, scale_chunk, out=output_chunk)

        return output.reshape(
            *prefix_shape, groups, packed_width * 2
        ).view(*prefix_shape, groups * packed_width * 2)
