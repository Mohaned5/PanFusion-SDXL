import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
import xformers.ops as xops


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        linear = nn.Linear(inner_dim, dim_out)
        linear.weight.data.fill_(0)
        linear.bias.data.fill_(0)
        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            linear
        )

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim

        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Linear(inner_dim, query_dim)
        self.to_out.weight.data.fill_(0)
        self.to_out.bias.data.fill_(0)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)

        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        mask = repeat(mask, 'b i j -> (b h) i j', h=h)
        # with torch.backends.cuda.sdp_kernel(enable_math=False, enable_mem_efficient=False):
        #     out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = xops.memory_efficient_attention(q, k, v, attn_bias=mask)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        return self.to_out(out)


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True)
                             for x in ctx.input_tensors]
        with torch.enable_grad(), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


class BasicTransformerBlock(nn.Module):

    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True,
                 disable_self_attn=False, use_checkpoint=True):
        super().__init__()

        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                                    context_dim=context_dim)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.checkpoint = checkpoint
        self.use_checkpoint=use_checkpoint

    def forward(self, x, context=None, mask=None, query_pe=None):
        if self. use_checkpoint:
            return checkpoint(self._forward, (x, context, mask, query_pe), self.parameters(), self.checkpoint)
        else:
            return self._forward(x, context, mask, query_pe)

    def _forward(self, x, context=None, mask=None, query_pe=None):
        if context is None:
            context = x
        query=x
        if query_pe is not None:
            query=query+query_pe
        query=self.norm1(query)
        context=self.norm1(context)
        x = self.attn1(query, context=context, mask=mask) + x
        x = self.ff(self.norm2(x)) + x

        return x


class SphericalPE(nn.Module):
    def __init__(self, N_freqs, logscale=True):
        """
        Defines a function that embeds input coordinates (e.g., theta, phi)
        to a vector of size N_freqs * 2.
        """
        super(SphericalPE, self).__init__()
        self.N_freqs = N_freqs

        if N_freqs <= 80:
            base = 2.0 
        else:
            base = 5000.0**(1.0 / (N_freqs / 2.5))
        if logscale:
            freq_bands = base**torch.linspace(0, N_freqs - 1, N_freqs)
        else:
            freq_bands = torch.linspace(1.0, 2.0**(N_freqs - 1), N_freqs)

        self.register_buffer('freq_bands', freq_bands)

    def forward(self, coords):
        """
        Embeds coords = (theta, phi) to a vector of size N_freqs*2 total dimension.

        Inputs:
            coords: (..., 2) where the last dimension holds (theta, phi)
        Outputs:
            out: (..., N_freqs * 2)
        """
        original_shape = coords.shape
        num_spatial_dims = len(original_shape) - 1
        target_feat_dim = self.N_freqs * 2
        target_shape = original_shape[:-1] + (target_feat_dim,)

        coords_flat = coords.reshape(-1, 2)
        coords_unsqueezed = coords_flat.unsqueeze(-1)

        encodings = coords_unsqueezed * self.freq_bands
        sin_encodings = torch.sin(encodings)  
        cos_encodings = torch.cos(encodings) 

        pe_concat = torch.cat([sin_encodings, cos_encodings], dim=-1)

        pe_combined = torch.mean(pe_concat, dim=1)
        if torch.isnan(pe_combined).any():
            raise ValueError("NaN found in positional encodings")

        output = pe_combined.reshape(target_shape)

        return output