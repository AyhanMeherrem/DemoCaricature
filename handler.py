import os
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DiffusionPipeline, EulerDiscreteScheduler
from diffusers.loaders import TextualInversionLoaderMixin
import safetensors.torch

from explicit_rome_attn_processors import ExplicitROMEAttnProcessor2_0, DefaultAttentionProcessor


class ExplicitROMEHandler:
    def __init__(self, pipeline: Union[DiffusionPipeline, TextualInversionLoaderMixin]):
        """
        A handler of Explicit ROME that utilizes Diffusers pipeline.

        Args:
            pipeline: an instance of DiffusionPipeline.
            pretrained_model_path: a path to a *directory* (for example `./my_id/`) containing weights.
        """

        self.pipeline = pipeline
        self.device = pipeline.device

        self.target_input = nn.Parameter(torch.zeros(self.pipeline.text_encoder.config.hidden_size)).to(device=self.pipeline.device)
        self.token = None
        self.token_index = None

    def load_explicit_rome(self, pretrained_model_path: str, token: str = "<ID>"):
        explicit_rome_path = os.path.join(pretrained_model_path, "explicit_rome.safetensors")
        explicit_rome_state_dict = safetensors.torch.load_file(explicit_rome_path, device="cpu")

        # load target input
        self.target_input.data = explicit_rome_state_dict["target_input"].to(device=self.pipeline.device)

        # load target outputs in all cross attention layers
        attn_procs = {}
        unet = self.pipeline.unet
        for i, name in enumerate(unet.attn_processors.keys()):
            is_cross_attention = 'attn2' in name
            if is_cross_attention:
                mod_name = name.replace(".processor", "")
                attn_proc_state_dict = {
                    "k_target_output": explicit_rome_state_dict[f"{mod_name}.k_target_output"].to(device=self.pipeline.device),
                    "v_target_output": explicit_rome_state_dict[f"{mod_name}.v_target_output"].to(device=self.pipeline.device),
                }
                cross_attention_dim = attn_proc_state_dict["k_target_output"].shape[0]
                attn_proc = ExplicitROMEAttnProcessor2_0(
                    cross_attention_dim=cross_attention_dim,
                )
                attn_proc.load_state_dict(attn_proc_state_dict)
                attn_proc = attn_proc.to(self.device)
                attn_procs[name] = attn_proc
            else:
                attn_procs[name] = DefaultAttentionProcessor()

        unet.set_attn_processor(attn_procs)

        # load placeholder indices
        embedding_path = os.path.join(pretrained_model_path, "embedding.safetensors")
        embedding_dict = safetensors.torch.load_file(embedding_path, device="cpu")
        self.pipeline.load_textual_inversion(embedding_dict, token=token)
        self.token = token
        self.token_index = self.pipeline.tokenizer.convert_tokens_to_ids(token)

    def encode_prompt(self, prompt, clip_skip):
        # textual inversion: process multi-vector tokens if necessary
        if isinstance(self.pipeline, TextualInversionLoaderMixin):
            prompt = self.pipeline.maybe_convert_prompt(prompt, self.pipeline.tokenizer)

        text_inputs = self.pipeline.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        if hasattr(self.pipeline.text_encoder.config, "use_attention_mask") and self.pipeline.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(self.pipeline.device)
        else:
            attention_mask = None

        if clip_skip is None:
            prompt_embeds = self.pipeline.text_encoder(text_input_ids.to(self.pipeline.device), attention_mask=attention_mask)
            prompt_embeds = prompt_embeds[0]
        else:
            prompt_embeds = self.pipeline.text_encoder(
                text_input_ids.to(self.pipeline.device), attention_mask=attention_mask, output_hidden_states=True
            )
            # Access the `hidden_states` first, that contains a tuple of
            # all the hidden states from the encoder layers. Then index into
            # the tuple to access the hidden states from the desired layer.
            prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
            # We also need to apply the final LayerNorm here to not mess with the
            # representations. The `last_hidden_states` that we typically use for
            # obtaining the final prompt representations passes through the LayerNorm
            # layer.
            prompt_embeds = self.pipeline.text_encoder.text_model.final_layer_norm(prompt_embeds)

        return prompt_embeds, text_input_ids

    def __call__(self, rome_scale=1., *args, **kwargs):
        if self.token is None:
            return self.pipeline(*args, **kwargs)

        prompt = kwargs.pop("prompt", None)
        guidance_scale = kwargs.pop("guidance_scale", 7.5)
        num_images_per_prompt = kwargs.pop("num_images_per_prompt", 1)
        clip_skip = kwargs.pop("clip_skip", None)

        prompt_embeds, text_input_ids = self.encode_prompt(prompt, clip_skip)
        target_similarity = F.cosine_similarity(prompt_embeds, self.target_input, dim=-1)
        target_similarity *= (text_input_ids == self.token_index).to(device=self.pipeline.device)
        target_similarity = target_similarity.to(dtype=self.pipeline.dtype)

        target_similarity *= rome_scale

        # only apply explicit ROME on positive prompt.
        # set similarity to zero on negative batches
        if guidance_scale > 1 and self.pipeline.unet.config.time_cond_proj_dim is None:
            target_similarity = torch.cat((torch.zeros_like(target_similarity), target_similarity), dim=0)

        cross_attention_kwargs = kwargs.pop("kwargs", dict())
        cross_attention_kwargs["target_similarity"] = target_similarity

        # set target_batch in cross_attention_kwargs

        return self.pipeline(
            prompt_embeds=prompt_embeds,
            guidance_scale=guidance_scale,
            cross_attention_kwargs=cross_attention_kwargs,
            num_images_per_prompt=num_images_per_prompt,
            clip_skip=clip_skip,
            **kwargs,
        )
