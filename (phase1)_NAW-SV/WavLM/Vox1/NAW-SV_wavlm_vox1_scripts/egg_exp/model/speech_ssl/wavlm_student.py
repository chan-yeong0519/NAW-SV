import math
from typing import Optional, Tuple, Union
import numpy as np

import torch
import torch.nn as nn
import transformers
from transformers import AutoConfig, WavLMModel
from transformers.modeling_outputs import BaseModelOutput, Wav2Vec2BaseModelOutput
import transformers.models.wavlm.modeling_wavlm as wavlm

BASE_PLUS = 'microsoft/wavlm-base-plus'



############
## Models ##
############
class Custom_WavLMPlus(nn.Module):
    def __init__(self, use_cls_token=False, adapter_hidden_size=None, mask_time_length=0, mask_time_prob=0.0, mask_feature_length=0, mask_feature_prob=0.0):
        super(Custom_WavLMPlus, self).__init__()
        
        # set transformer encoder
        self.config = AutoConfig.from_pretrained(BASE_PLUS)
        self.config.mask_time_length = mask_time_length
        self.config.mask_time_prob = mask_time_prob
        self.config.mask_feature_length = mask_feature_length
        self.config.mask_feature_prob = mask_feature_prob
        self.wavlm = CustomWavLMModel(config=self.config, use_cls_token=use_cls_token, adapter_hidden_size=adapter_hidden_size)
        
        # weight initialization
        ssl = WavLMModel.from_pretrained(
            BASE_PLUS,
            from_tf=bool(".ckpt" in BASE_PLUS),
            config=AutoConfig.from_pretrained(BASE_PLUS),
            revision="main",
            ignore_mismatched_sizes=False,
        )
        self.wavlm.feature_extractor.load_state_dict(ssl.feature_extractor.state_dict(), strict=False)
        self.wavlm.feature_projection.load_state_dict(ssl.feature_projection.state_dict(), strict=False)
        for i in range(self.config.num_hidden_layers):
            self.wavlm.encoder.layers[i].load_state_dict(ssl.encoder.layers[i].state_dict(), strict=False)
        
    def forward(self, x, output_hidden_states=True, idx_without_adapter=None, masking=False):
        x = self.wavlm(x, output_hidden_states=output_hidden_states, idx_without_adapter=idx_without_adapter, masking=masking)
        
        return x

class CustomWavLMModel(WavLMModel):
    def __init__(self, config, use_cls_token=False, adapter_hidden_size=None):
        super().__init__(config)
        self.config = config
        self.feature_extractor = wavlm.WavLMFeatureEncoder(config)
        self.feature_projection = wavlm.WavLMFeatureProjection(config)
        self.use_cls_token = use_cls_token
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.rand(1, config.hidden_size), requires_grad=True) # CLS token
        
        # model only needs masking vector if mask prob is > 0.0
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        if config.do_stable_layer_norm:
            self.encoder = CustomWavLMEncoderStableLayerNorm(config, adapter_hidden_size=adapter_hidden_size)
        else:
            self.encoder = CustomWavLMEncoder(config, adapter_hidden_size=adapter_hidden_size)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        idx_without_adapter: Optional[list] = None,
        masking: Optional[bool] = None,
    ) -> Union[Tuple, transformers.modeling_outputs.Wav2Vec2BaseModelOutput]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # CNN
        extract_features = self.feature_extractor(input_values)
        
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)

        hidden_states, extract_features = self.feature_projection(extract_features)
        
        # add CLS token
        if self.use_cls_token:
            hidden_states = torch.cat((self.cls_token.repeat(hidden_states.size(0), 1, 1), hidden_states), dim=1)

        hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask, masking=masking)

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            idx_without_adapter=idx_without_adapter,
        )

        hidden_states = encoder_outputs[0]

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
    
    def _mask_hidden_states(
        self,
        hidden_states: torch.FloatTensor,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        masking: Optional[bool] = None,
    ):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """

        # `config.apply_spec_augment` can set masking to False
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states

        # generate indices & apply SpecAugment along time axis
        batch_size, sequence_length, hidden_size = hidden_states.size()

        if mask_time_indices is not None:
            # apply SpecAugment along time axis with given mask_time_indices
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        elif self.config.mask_time_prob > 0 and masking:
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.config.mask_time_min_masks,
            )
            mask_time_indices = torch.tensor(mask_time_indices, device=hidden_states.device, dtype=torch.bool)
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)

        if self.config.mask_feature_prob > 0 and masking:
            # generate indices & apply SpecAugment along feature axis
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            )
            mask_feature_indices = torch.tensor(mask_feature_indices, device=hidden_states.device, dtype=torch.bool)
            mask_feature_indices = mask_feature_indices[:, None].expand(-1, sequence_length, -1)
            hidden_states[mask_feature_indices] = 0

        return hidden_states
     
class CustomWavLMEncoder(nn.Module):
    def __init__(self, config, adapter_hidden_size=None):
        super().__init__()
        self.config = config
        self.pos_conv_embed = wavlm.WavLMPositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(
            [CustomWavLMEncoderLayer(config, has_relative_position_bias=(i == 0), adapter_hidden_size=adapter_hidden_size) for i in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        idx_without_adapter=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # make sure padded tokens output 0
            hidden_states[~attention_mask] = 0.0

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        deepspeed_zero3_is_enabled = transformers.deepspeed.is_deepspeed_zero3_enabled()
        position_bias = None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = np.random.uniform(0, 1)

            skip_the_layer = self.training and i > 0 and (dropout_probability < self.config.layerdrop)
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # under deepspeed zero3 all gpus must run in sync
                if self.gradient_checkpointing and self.training:
                    # create gradient checkpointing function
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer),
                        hidden_states,
                        attention_mask,
                        position_bias,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_bias=position_bias,
                        output_attentions=output_attentions,
                        index=i,
                        idx_without_adapter=idx_without_adapter,
                    )

                hidden_states, position_bias = layer_outputs[:2]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

class CustomWavLMEncoderStableLayerNorm(nn.Module):
    def __init__(self, config, adapter_hidden_size=None):
        super().__init__()
        self.config = config
        self.pos_conv_embed = wavlm.WavLMPositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(
            [
                CustomWavLMEncoderLayerStableLayerNorm(config, has_relative_position_bias=(i == 0), adapter_hidden_size=adapter_hidden_size)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        idx_without_adapter=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # make sure padded tokens are not attended to
            hidden_states[~attention_mask] = 0

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.dropout(hidden_states)

        deepspeed_zero3_is_enabled = transformers.deepspeed.is_deepspeed_zero3_enabled()
        position_bias = None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = np.random.uniform(0, 1)

            skip_the_layer = self.training and i > 0 and (dropout_probability < self.config.layerdrop)
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # under deepspeed zero3 all gpus must run in sync
                # XXX: could optimize this like synced_gpus in generate_utils but not sure if it's worth the code complication
                if self.gradient_checkpointing and self.training:
                    # create gradient checkpointing function
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer),
                        hidden_states,
                        attention_mask,
                        position_bias,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        output_attentions=output_attentions,
                        position_bias=position_bias,
                        idx_without_adapter=idx_without_adapter,
                    )
                hidden_states, position_bias = layer_outputs[:2]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[2],)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, 
            hidden_states=all_hidden_states, 
            attentions=all_self_attentions
        )

class CustomWavLMEncoderLayer(nn.Module):
    def __init__(self, config, has_relative_position_bias: bool = True, adapter_hidden_size=None):
        super().__init__()
        self.attention = wavlm.WavLMAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            num_buckets=config.num_buckets,
            max_distance=config.max_bucket_distance,
            has_relative_position_bias=has_relative_position_bias,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = wavlm.WavLMFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.use_adapter = adapter_hidden_size is not None
        if self.use_adapter:
            self.adapter = WavLMAdapter(config.hidden_size, adapter_hidden_size)
            
    def forward(self, hidden_states, attention_mask=None, position_bias=None, output_attentions=False, index=0, idx_without_adapter=None):
        attn_residual = hidden_states
        hidden_states, attn_weights, position_bias = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
            index=index,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        hidden_states = self.layer_norm(hidden_states)

        if self.use_adapter:
            if idx_without_adapter is None:
                adapt_hidden_states = self.adapter(hidden_states)
                hidden_states = hidden_states + self.feed_forward(hidden_states) + adapt_hidden_states
            else:
                # separate branch (clean & noisy)
                ff_hidden_states = self.feed_forward(hidden_states)
                clean_hidden_states, noisy_hidden_states = hidden_states[:idx_without_adapter, :, :], hidden_states[idx_without_adapter:, :, :]
                clean_ff_hidden_states, noisy_ff_hidden_states = ff_hidden_states[:idx_without_adapter, :, :], ff_hidden_states[idx_without_adapter:, :, :]

                # clean batch
                clean_hidden_states = clean_hidden_states + clean_ff_hidden_states
                
                # noisy batch
                adapt_hidden_states = self.adapter(noisy_hidden_states)
                noisy_hidden_states = noisy_hidden_states + noisy_ff_hidden_states + adapt_hidden_states
                
                # merge
                hidden_states = torch.cat((clean_hidden_states, noisy_hidden_states), dim=0)

        else:
            hidden_states = hidden_states + self.feed_forward(hidden_states)

        hidden_states = self.final_layer_norm(hidden_states)
        
        outputs = (hidden_states, position_bias)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

class CustomWavLMEncoderLayerStableLayerNorm(nn.Module):
    def __init__(self, config, has_relative_position_bias: bool = True, adapter_hidden_size=None):
        super().__init__()
        self.attention = wavlm.WavLMAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            num_buckets=config.num_buckets,
            max_distance=config.max_bucket_distance,
            has_relative_position_bias=has_relative_position_bias,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = wavlm.WavLMFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.use_adapter = adapter_hidden_size is not None
        if self.use_adapter:
            self.adapter = WavLMAdapter(config.hidden_size, adapter_hidden_size)
            
    def forward(self, hidden_states, attention_mask=None, position_bias=None, output_attentions=False, idx_without_adapter=None):
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attn_weights, position_bias = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        
        if self.use_adapter:
            if idx_without_adapter is None:
                hidden_states = self.final_layer_norm(hidden_states)
                adapt_hidden_states = self.adapter(hidden_states)
                hidden_states = hidden_states + self.feed_forward(hidden_states) + adapt_hidden_states
            else:
                # separate branch (clean & noisy)
                ff_hidden_states = self.feed_forward(self.final_layer_norm(hidden_states))
                clean_hidden_states, noisy_hidden_states = hidden_states[:idx_without_adapter, :, :], hidden_states[idx_without_adapter:, :, :]
                clean_ff_hidden_states, noisy_ff_hidden_states = ff_hidden_states[:idx_without_adapter, :, :], ff_hidden_states[idx_without_adapter:, :, :]

                # clean batch
                clean_hidden_states = clean_hidden_states + clean_ff_hidden_states
                
                # noisy batch
                adapt_hidden_states = self.adapter(noisy_hidden_states)
                noisy_hidden_states = noisy_hidden_states + noisy_ff_hidden_states + adapt_hidden_states
                
                # merge
                hidden_states = torch.cat((clean_hidden_states, noisy_hidden_states), dim=0)
        else:
            hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))

        outputs = (hidden_states, position_bias)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

class WavLMAdapter(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()

        self.down = nn.Linear(in_channels, hidden_channels)
        self.elu = nn.ELU()
        self.up = nn.Linear(hidden_channels, in_channels)
        
    def forward(self, x):
        # projection
        x = self.down(x)
        x = self.elu(x)
        x = self.up(x)
        
        return x

def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[torch.LongTensor] = None,
    min_masks: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape. Used to implement [SpecAugment: A Simple Data Augmentation Method for
    ASR](https://arxiv.org/abs/1904.08779). Note that this method is not optimized to run on TPU and should be run on
    CPU as part of the preprocessing during training.

    Args:
        shape: The shape for which to compute masks. This should be of a tuple of size 2 where
               the first element is the batch size and the second element is the length of the axis to span.
        mask_prob:  The percentage of the whole axis (between 0 and 1) which will be masked. The number of
                    independently generated mask spans of length `mask_length` is computed by
                    `mask_prob*shape[1]/mask_length`. Note that due to overlaps, `mask_prob` is an upper bound and the
                    actual percentage will be smaller.
        mask_length: size of the mask
        min_masks: minimum number of masked spans
        attention_mask: A (right-padded) attention mask which independently shortens the feature axis of
                        each batch dimension.
    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )

    # epsilon is used for probabilistic rounding
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)

        # make sure num masked span <= sequence_length
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # make sure num_masked span is also <= input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # compute number of masked spans in batch
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )

    # SpecAugment mask to fill
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = []

    max_num_masked_span = compute_num_masked_span(sequence_length)

    if max_num_masked_span == 0:
        return spec_aug_mask

    for input_length in input_lengths:
        # compute num of masked spans for this input
        num_masked_span = compute_num_masked_span(input_length)

        # get random indices to mask
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # pick first sampled index that will serve as a dummy index to pad vector
        # to ensure same dimension for all batches due to probabilistic rounding
        # Picking first sample just pads those vectors twice.
        if len(spec_aug_mask_idx) == 0:
            # this case can only happen if `input_length` is strictly smaller then
            # `sequence_length` in which case the last token has to be a padding
            # token which we can use as a dummy mask id
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # expand masked indices to masked spans
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # add offset to the starting indexes so that indexes now create a span
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # ensure that we cannot have indices larger than sequence_length
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # scatter indices to mask
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    return spec_aug_mask