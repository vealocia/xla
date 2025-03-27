import sys
import os
example_folder = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(example_folder)
import decoder_only_model
from train_decoder_only_base import TrainDecoderOnlyBase

import functools

import torch
import numpy as np
import torch_xla.distributed.spmd as xs
import torch_xla.utils.utils as xu
import torch_xla.distributed.parallel_loader as pl
from torch_xla.experimental.spmd_fully_sharded_data_parallel import SpmdFullyShardedDataParallel as FSDPv2
from torch_xla import runtime as xr
from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy

# checkout our doc at https://github.com/pytorch/xla/blob/master/docs/fsdpv2.md
class TrainDecoderOnlyFSDPv2(TrainDecoderOnlyBase):

  def __init__(self):
    super().__init__()
    # Define the mesh following common SPMD practice
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (num_devices, 1)
    device_ids = np.array(range(num_devices))
    # To be noted, the mesh must have an axis named 'fsdp', which the weights and activations will be sharded on.
    mesh = xs.Mesh(device_ids, mesh_shape, ('fsdp', 'model'))
    xs.set_global_mesh(mesh)

    # Shard the input(data parallel).
    # Scale the batch size with num_devices since there will be only one
    # process that handles all runtime devices.
    self.batch_size *= num_devices
    train_loader = xu.SampleGenerator(
        data=(torch.zeros(self.batch_size, self.seq_len, dtype=torch.int64),
              torch.zeros(self.batch_size, self.seq_len, dtype=torch.int64)),
        sample_count=self.train_dataset_len // self.batch_size)
    print("global batch size:", self.batch_size)
    self.train_device_loader = pl.MpDeviceLoader(
        train_loader,
        self.device,
        # Shard the input's batch dimension along the `fsdp` axis, no sharding along other dimensions
        input_sharding=xs.ShardingSpec(mesh, ('fsdp', None)))

    # Apply FSDP sharding on each DecoderLayer layer.
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2DecoderLayer, repeat_kv
    from torch_xla.experimental.custom_kernel import flash_attention
    
    def torchxla_flash_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask=None,
        dropout=0.0,
        scaling=None,
        sliding_windows=None,
        **kwargs,
    ):
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)

        attn_output = flash_attention(
            query, key_states, value_states, causal=True,
            sm_scale=scaling,
            partition_spec=('fsdp', None, None, None))
        print(query.shape, key_states.shape, value_states.shape, attn_output.shape)

        return attn_output, None
    
    ALL_ATTENTION_FUNCTIONS.update(
        {
            "torchxla_flash_attention": torchxla_flash_attention_forward,
        }
    )
    
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            Qwen2DecoderLayer,
        },
    )

    self.model = Qwen2ForCausalLM.from_pretrained("/mnt/disks/cambrian_s/checkpoints/Qwen2-7B-Instruct")
    del self.model.model.layers[16:]
    for layer in self.model.model.layers:
        layer.self_attn.config._attn_implementation = "torchxla_flash_attention"
        # layer.self_attn.config._attn_implementation = "sdpa"
    self.model.to(torch.bfloat16)

    print(self.model)
    print("num_parameters:", sum(p.numel() for p in self.model.parameters()))

    from torch_xla.distributed.fsdp import checkpoint_module
    def auto_wrapper_callable(m, *args, **kwargs):
        target_cls = FSDPv2
        return target_cls(checkpoint_module(m), *args, **kwargs)

    # FSDPv2 will use the global mesh set above
    self.model = FSDPv2(
        self.model,
        auto_wrap_policy=auto_wrap_policy,
        auto_wrapper_callable=auto_wrapper_callable,
        )
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0)


if __name__ == '__main__':
  # Enable the SPMD
  xr.use_spmd()
  base = TrainDecoderOnlyFSDPv2()
  base.start_training()
