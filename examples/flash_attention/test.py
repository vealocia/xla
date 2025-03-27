import torch
import torch.nn.functional as F
import torch_xla
import torch_xla.core.xla_model as xm

from torch_xla.experimental.custom_kernel import flash_attention

scale = 1.0 / 8.0
for _ in range(10):
    q = torch.randn(1, 16, 1024, 64, dtype=torch.bfloat16).to(xm.xla_device())
    k = torch.randn(1, 16, 1024, 64, dtype=torch.bfloat16).to(xm.xla_device())
    v = torch.randn(1, 16, 1024, 64, dtype=torch.bfloat16).to(xm.xla_device())

    output_torch = F.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=True)
    xm.mark_step()
    # output = torch_xla.experimental.custom_kernel.flash_attention(q, k, v, sm_scale=scale)
    output = flash_attention(q, k, v, sm_scale=scale, causal=True)
    xm.mark_step()
    try:
        torch.testing.assert_close(output, output_torch)
    except AssertionError as e:
        print(e, "\n\n")
