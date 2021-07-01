import torch
from torch import nn
from torch.nn import Module
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init

import math
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
        self.hidden_size = 1024
        self.num_attention_heads = 16
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.num_layers = 4

class BertTest(nn.Module):
    def __init__(self, config):
        super(BertTest, self).__init__()
        self.layers = nn.ModuleList([Fusion(config) for x in range(config.num_layers)]) 

    def forward(self, input_tensor, attention_mask):
        my_input_tensor = input_tensor
        for layer in self.layers :
            output_tensor = layer(my_input_tensor, attention_mask)
            my_input_tensor = output_tensor
        return output_tensor

class Fusion(nn.Module):
    def __init__(self, config):
        super(Fusion, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        x = x.view(x.size(0), x.size(1) * self.num_attention_heads, self.attention_head_size).transpose(0,1)
        return x

    def transpose_key_for_scores(self, x):
        x = x.view(x.size(0), x.size(1) * self.num_attention_heads, self.attention_head_size).transpose(0,1).transpose(1,2)
        return x

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_key_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.bmm(query_layer, key_layer)
        attention_scores = attention_scores.view(int(attention_scores.size(0) / self.num_attention_heads),
                                                 self.num_attention_heads,
                                                 attention_scores.size(1),
                                                 attention_scores.size(2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        attention_probs = attention_probs.view(attention_probs.size(0)*attention_probs.size(1), attention_probs.size(2), attention_probs.size(3))

        context_layer = torch.bmm(attention_probs, value_layer)

        context_layer = context_layer.transpose(0,1).contiguous()
        context_layer = context_layer.view(context_layer.size(0), int(context_layer.size(1) / self.num_attention_heads), self.all_head_size)
        return context_layer

class BertLayerNorm(Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.layer_norm = torch.nn.LayerNorm(hidden_size, eps)

    def forward(self, x):
        return self.layer_norm(x)


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        output1 = self.dense(hidden_states)
        output2 = self.dropout(output1)
        output3 = output2 + input_tensor
        output4 = self.LayerNorm(output3)
        return output4

if __name__ == "__main__" :
    device = 'xla'
    inputs = torch.randn(8, 512, 1024, device=device, dtype=torch.float, requires_grad=True).transpose(0,1)
    mask = torch.randn(8, 1, 1, 512, device=device, dtype=torch.float, requires_grad=False)
    mask_bool = mask > 0.
    grads = torch.randn(512, 8, 1024, device=device, dtype=torch.float, requires_grad=False)

    ltm.mark_step()
    model = Fusion(BertConfig())
    model = model.to(device='xla')

    for idx in range(1) :
        out = model(inputs, mask_bool)
        out.backward(grads)
        ltm.mark_step()

    print(metrics.metrics_report())
