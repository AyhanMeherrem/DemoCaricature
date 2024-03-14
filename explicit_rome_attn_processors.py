import torch
import torch.nn as nn
from torch.nn import functional as F
from diffusers.models.attention_processor import AttnProcessor2_0, Attention


class DefaultAttentionProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.processor = AttnProcessor2_0()

    def __call__(
        self, 
        attn: Attention, 
        hidden_states: torch.Tensor, 
        encoder_hidden_states=None,
        attention_mask=None,
        target_similarity=None,  # disable logger warning
        **kwargs,
    ):
        return self.processor(attn, hidden_states, encoder_hidden_states, attention_mask)


class ExplicitROMEAttnProcessor2_0(nn.Module):
    def __init__(
        self,
        cross_attention_dim: int,
    ):
        """
        Processor for implementing the Explicit ROME attention mechanism.

        Args:
            cross_attention_dim (`int`):
                The number of channels in the `encoder_hidden_states`.
        """

        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.cross_attention_dim = cross_attention_dim

        self.k_target_output = nn.Parameter(torch.zeros(self.cross_attention_dim))
        self.v_target_output = nn.Parameter(torch.zeros(self.cross_attention_dim))

    def __call__(
            self,
            attn: Attention,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            attention_mask: torch.Tensor = None,
            target_similarity: torch.Tensor = None,
            *args,
            **kwargs
    ):
        """
        Args:
            target_similarity:
                The similarity used in explicit ROME, value of entries irrelevant custom concept will be 0 .
                shape: (B, L)
        """

        if target_similarity is None or self.k_target_output is None or self.v_target_output is None:
            attn._modules.pop("processor")
            attn.processor = AttnProcessor2_0()
            return attn.processor(attn, hidden_states, *args, **kwargs)

        residual = hidden_states

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        # Explicit ROME
        key = key + (self.k_target_output[None, None, :] * target_similarity[..., None]).to(dtype=key.dtype)
        value = value + (self.v_target_output[None, None, :] * target_similarity[..., None]).to(dtype=key.dtype)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
