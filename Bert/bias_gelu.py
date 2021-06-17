import torch
from torch.nn import Module
import torch.nn.functional as F
import lazy_tensor_core.core.lazy_model as ltm
import lazy_tensor_core.debug.metrics as metrics
import lazy_tensor_core

lazy_tensor_core._LAZYC._ltc_init_ts_backend()
torch._C._jit_set_nvfuser_enabled(True)
torch._C._jit_set_texpr_fuser_enabled(False)
torch._C._jit_set_profiling_executor(True)
torch._C._jit_set_profiling_mode(True)
torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_set_bailout_depth(20)

class BertConfig :
    def __init__(self) :
        self.hidden_size = 4096 
        self.num_attention_heads = 16
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.num_layers = 10

class Fusion(Module):
    def __init__(self, config, device):
        super(Fusion, self).__init__()
        self.bias = torch.nn.Parameter(torch.zeros(config.hidden_size, device=device))

    def forward(self, inputs):
        out1 = inputs + self.bias
        out2 = F.gelu(out1)
        return out2

if __name__ == "__main__" :
    device = 'xla'
    inputs = torch.randn(8, 512, 4096, device=device, dtype=torch.float, requires_grad=True)
    grads = torch.randn(8, 512, 4096, device=device, dtype=torch.float, requires_grad=False)

    ltm.mark_step()

    model = Fusion(BertConfig(), device=device)

    for idx in range(1) :
        out = model(inputs)
        out.backward(grads)
        ltm.mark_step()

    print(metrics.metrics_report())
