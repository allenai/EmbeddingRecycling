"""SkipBERT modeling"""

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import transformers
from transformers import BertPreTrainedModel, BertModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler, BertLayer
from transformers.models.bert.modeling_bert import BertPreTrainingHeads
from transformers.modeling_outputs import SequenceClassifierOutput
from . import plot

import logging
logger = logging.getLogger(__name__)

logger.warn('Hacking BertSelfAttention! Now it returns attention scores rather than probabilities.')

class BertSelfAttention(transformers.models.bert.modeling_bert.BertSelfAttention):

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):

        device = hidden_states.device
        mixed_query_layer = self.query(hidden_states)

        # most codes are copied from transformers v4.3.3
        
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores.to(device) + attention_mask.to(device)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
            #attention_scores = attention_scores * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_scores) if output_attentions else (context_layer,) # hacked: replace attention_probs with attention_scores

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs
    
transformers.models.bert.modeling_bert.BertSelfAttention = BertSelfAttention



class BertForPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        fit_size = getattr(config, 'fit_size', 768)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)
        self.fit_denses = nn.ModuleList(
            [nn.Linear(config.hidden_size, fit_size) for _ in range(config.num_hidden_layers+1)]
        )

    def forward(self, input_ids, token_type_ids=None,
                attention_mask=None, masked_lm_labels=None,
                next_sentence_label=None, labels=None,
                output_attentions=True, output_hidden_states=True,):
        outputs = self.bert(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        sequence_output, att_output, pooled_output = outputs.hidden_states, outputs.attentions, outputs.pooler_output
        tmp = []
        for s_id, sequence_layer in enumerate(sequence_output):
            tmp.append(self.fit_denses[s_id](sequence_layer))
        sequence_output = tmp

        return att_output, sequence_output
    
    
    

class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, do_fit=False, share_param=True):
        super().__init__(config)
        num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        
        self.do_fit, self.share_param = do_fit, share_param
        if self.do_fit:
            fit_size = getattr(config, 'fit_size', 768)
            self.fit_size = fit_size
            if self.share_param:
                self.fit_dense = nn.Linear(config.hidden_size, fit_size)
            else:
                self.fit_denses = nn.ModuleList(
                    [nn.Linear(config.hidden_size, fit_size) for _ in range(config.num_hidden_layers + 1)]
                )

    def do_fit_dense(self, sequence_output):
        
        tmp = []
        if self.do_fit:
            for s_id, sequence_layer in enumerate(sequence_output):
                if self.share_param:
                    tmp.append(self.fit_dense(sequence_layer))
                else:
                    tmp.append(self.fit_denses[s_id](sequence_layer))
            sequence_output = tmp
            
        return sequence_output

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):

        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                            output_hidden_states=True, output_attentions=True)
        sequence_output, att_output, pooled_output = outputs.hidden_states, outputs.attentions, outputs.pooler_output
        
        logits = self.classifier(pooled_output)
        
        sequence_output = self.do_fit_dense(sequence_output)

        return logits, att_output, sequence_output
    
    
    
class BertForSequenceClassificationPrediction(BertForSequenceClassification):
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        
        assert not self.training
        
        _, pooled_output, sequence_output, att_output = self.bert(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
            output_hidden_states=True, output_attentions=True)
        
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss = torch.tensor(0.)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )
    

class ShallowSkipping(nn.Module):
    
    def __init__(self, model):
        super().__init__()
#         self.model = model # do not register
        self.config = model.config
        self.shallow_config = model.shallow_config
        # current only support trigram
        self.ngram = 3
        self.config.max_num_entries = 10000000
        
        if self.shallow_config.hidden_size != self.config.hidden_size:
            self.linear = nn.Linear(self.shallow_config.hidden_size, self.config.hidden_size)
            
        self.plot = plot.Plot(self.config.max_num_entries, self.config.hidden_size)
        
    def _build_tri_gram_ids(self, input_ids:torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(
            self.plot.input_ids_to_tri_grams(input_ids.cpu().numpy())
        ).to(input_ids.device)
        
    def build_input_ngrams(self, input_ids:torch.Tensor, token_type_ids:torch.Tensor):
        
        input_ngram_ids = self._build_tri_gram_ids(input_ids)
        
        token_ngram_type_ids = None #
        
        attention_mask = (input_ngram_ids > 0).float()

        self.config.ngram_masking = 0.0
        
        if self.training:
            _mask = torch.rand(attention_mask.shape).to(attention_mask.device)
            _mask = (_mask > self.config.ngram_masking)
            attention_mask *= _mask

        attention_mask[:, self.ngram//2] = 1 # avoid masking all tokens in a tri-gram
        return input_ngram_ids, token_ngram_type_ids, attention_mask
    
    @torch.jit.script
    def merge_ngrams(input_ids, ngram_hidden_states, aux_embeddings):
        batch_size, seq_length = input_ids.shape
        lens = (input_ids!=0).sum(1)
        hidden_state = torch.zeros([batch_size, seq_length, ngram_hidden_states.size(-1)], dtype=ngram_hidden_states.dtype, device=ngram_hidden_states.device)
        
        # assert to be trigrams
        flat_hidden_state = ngram_hidden_states[:, 1]
        flat_hidden_state[:-1] = flat_hidden_state[:-1] + ngram_hidden_states[1:, 0]
        flat_hidden_state[1:] = flat_hidden_state[1:] + ngram_hidden_states[:-1, 2]
        k = 0
        for i in range(batch_size):
            hidden_state[i, :lens[i]] = flat_hidden_state[k: k+lens[i]]
            k += 1 + lens[i] # 1 for skipping one padding tri-gram
        hidden_state = hidden_state + aux_embeddings
        return hidden_state
    
    def forward_shallow_layers(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        ngram_mask_position=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=True,
        output_hidden_states=True,
        model=None,
    ):
        device = model.device
        
        input_ngram_ids, token_ngram_type_ids, attention_mask = self.build_input_ngrams(input_ids, token_type_ids)
        ngram_attention_mask = attention_mask.clone()
        
        if ngram_mask_position is not None:
            input_ngram_ids[:, ngram_mask_position] = 0
            ngram_attention_mask[:, ngram_mask_position] = 0

        extended_attention_mask = model.get_extended_attention_mask(attention_mask, input_ngram_ids.shape, device)

        ngram_index=(input_ngram_ids[:, self.ngram//2] > 0)

        embedding_output = model.embeddings(input_ids=input_ngram_ids, token_type_ids=token_ngram_type_ids)

        hidden_states = embedding_output
        attention_mask = extended_attention_mask

        for i, layer_module in enumerate(
                model.encoder.layer[:self.config.num_hidden_layers - self.config.num_full_hidden_layers]):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=layer_head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]
            
        if self.shallow_config.hidden_size != self.config.hidden_size:
            hidden_states = self.linear(hidden_states)
            
        # Set zero the padding ngrams: (..., [PAD], ...)
        hidden_states = hidden_states * ngram_index[:, None, None]
            
        hidden_states = hidden_states * model.attn(hidden_states).sigmoid() * ngram_attention_mask.unsqueeze(-1)
        
        return input_ngram_ids, hidden_states

    def forward(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=True,
        output_hidden_states=True,
        model=None,
    ):
        
        device = model.device
        
        batch_size, seq_length = input_ids.shape
        aux_embeddings = model.embeddings.position_embeddings2.weight[:seq_length].unsqueeze(0)
        aux_embeddings = aux_embeddings + model.embeddings.token_type_embeddings2(token_type_ids)

        self.config.plot_mode = 'force_compute'
        
        if self.config.plot_mode == 'force_compute':
            '''
            compute only, ignore PLOT
            '''
            input_ngram_ids, hidden_states = self.forward_shallow_layers(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                ngram_mask_position=None,
                model=model,
            )
            
        elif self.config.plot_mode == 'update_all':
            '''
            build PLOT
            '''
            # uni-grams
            input_ngram_ids, hidden_states = self.forward_shallow_layers(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                ngram_mask_position=(0,2),
                model=model,
            )
            self.plot.update_data(input_ngram_ids, hidden_states)
            
            # bi-grams
            input_ngram_ids, hidden_states = self.forward_shallow_layers(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                ngram_mask_position=0,
                model=model,
            )
            self.plot.update_data(input_ngram_ids, hidden_states)
            
            # tri-grams
            input_ngram_ids, hidden_states = self.forward_shallow_layers(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                ngram_mask_position=None,
                model=model,
            )
            self.plot.update_data(input_ngram_ids, hidden_states)
            
        elif self.config.plot_mode == 'plot_passive':
            '''
            use plot if no oov
            '''
            
            if input_ids.is_cuda:
                input_ids = input_ids.cpu()
            if not self.plot.has_oov(input_ids):
                hidden_states = self.plot.retrieve_data(input_ids)
                hidden_states = hidden_states.to(device)
            else:
                input_ids = input_ids.to(device)
                input_ngram_ids, hidden_states = self.forward_shallow_layers(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    ngram_mask_position=None,
                    model=model,
                )
                self.plot.update_data(input_ngram_ids, hidden_states)
                
        elif self.config.plot_mode == 'plot_only':
            '''
            plot only
            looking up order: trigram -> bigram -> unigram -> 0
            '''
            if input_ids.is_cuda:
                logger.warn("'input_ids' is better to placed in CPU.")
                input_ids = input_ids.cpu()
            hidden_states = self.plot.retrieve_data(input_ids)
            hidden_states = hidden_states.to(device)
                
        hidden_states = F.dropout(hidden_states, self.config.hidden_dropout_prob, self.training)
        hidden_states = self.merge_ngrams(input_ids, hidden_states, aux_embeddings)
        hidden_states = model.norm(hidden_states)

        return hidden_states
    
    
class SkipBertEncoder(BertEncoder):
    def __init__(self, shallow_config, config):
        super(BertEncoder, self).__init__()
        self.config = config
        self.shallow_config = shallow_config
        config.num_full_hidden_layers = 6
        self.layer = nn.ModuleList(
            [
                BertLayer(shallow_config) for _ in range(config.num_hidden_layers - config.num_full_hidden_layers)
            ] + [
                BertLayer(config) for _ in range(config.num_full_hidden_layers)
            ])
    
class SkipBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.shallow_config = copy.deepcopy(config)
        
        self.shallow_config.hidden_size = getattr(config, 'shallow_hidden_size', 768)
        self.shallow_config.intermediate_size = getattr(config, 'shallow_intermediate_size', 3072)

        self.embeddings = BertEmbeddings(self.shallow_config)
        self.encoder = SkipBertEncoder(self.shallow_config, config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.embeddings.position_embeddings2 = nn.Embedding(self.config.max_position_embeddings, self.config.hidden_size)
        self.embeddings.token_type_embeddings2 = nn.Embedding(self.config.type_vocab_size, self.config.hidden_size)
        
        self.norm = nn.LayerNorm(self.config.hidden_size)
        self.attn = nn.Linear(self.config.hidden_size, 1)
        self.shallow_skipping = ShallowSkipping(self)
        
        self.init_weights()
            
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=True,
        output_hidden_states=True,
    ):
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        input_shape = input_ids.size()
        device = self.device

        if attention_mask is None:
            attention_mask = (input_ids != 0).float()
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        hidden_states = self.shallow_skipping(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            model=self,
        )

        # Global transformer layers
        attention_mask = extended_attention_mask

        all_hidden_states = ()
        all_self_attentions = ()

        for i, layer_module in enumerate(self.encoder.layer[-self.config.num_full_hidden_layers:]):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i + self.config.num_hidden_layers - self.config.num_full_hidden_layers] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=layer_head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=None,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        sequence_output = hidden_states
        pooled_output = self.pooler(sequence_output)
        
        return (sequence_output, pooled_output, all_hidden_states, all_self_attentions)
    
    def freeze_shallow_layers(self):
        for p in self.embeddings.parameters():
            p.requires_grad = False
        for layer in self.encoder.layer[:self.config.num_hidden_layers - self.config.num_full_hidden_layers]:
            for p in layer.parameters():
                p.requires_grad = False
        try:
            for p in self.shallow_skipping.linear.parameters():
                p.requires_grad = False
        except Exception as e:
            pass
        try:
            for p in self.attn.parameters():
                p.requires_grad = False
        except Exception as e:
            pass
                
        self.embeddings.dropout.p = 0.
        for layer in self.encoder.layer[:self.config.num_hidden_layers - self.config.num_full_hidden_layers]:
            for m in layer.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.p = 0.
    

class SkipBertForPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.fit_size = getattr(config, 'fit_size', 768)
        self.bert = SkipBertModel(config)
        self.cls = BertPreTrainingHeads(config)
        
        if self.fit_size != config.hidden_size:
            self.fit_denses = nn.ModuleList(
                [nn.Linear(config.hidden_size, self.fit_size) for _ in range(config.num_hidden_layers + 1)]
            )

    def forward(self, input_ids, token_type_ids=None,
                attention_mask=None, masked_lm_labels=None,
                next_sentence_label=None, labels=None,
                output_attentions=True, output_hidden_states=True,):
        _, pooled_output, sequence_output, att_output = self.bert(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        
        if self.fit_size != self.config.hidden_size:
            tmp = []
            for s_id, sequence_layer in enumerate(sequence_output):
                tmp.append(self.fit_denses[s_id](sequence_layer))
            sequence_output = tmp

        return att_output, sequence_output

    
    def freeze_shallow_layers(self):
        self.bert.freeze_shallow_layers()

    
class SkipBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, do_fit=False, share_param=True):
        super().__init__(config)
        num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.num_labels = num_labels
        self.bert = SkipBertModel(config)
        self.dropout = nn.Dropout(
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        
        self.do_fit, self.share_param = do_fit, share_param
        if self.do_fit:
            fit_size = getattr(config, 'fit_size', 768)
            self.fit_size = fit_size
            if self.share_param:
                self.share_fit_dense = nn.Linear(config.hidden_size, fit_size)
            else:
                self.fit_denses = nn.ModuleList(
                    [nn.Linear(config.hidden_size, fit_size) for _ in range(config.num_hidden_layers + 1)]
                )

    def do_fit_dense(self, sequence_output):
        
        tmp = []
        if self.do_fit:
            for s_id, sequence_layer in enumerate(sequence_output):
                if self.share_param:
                    tmp.append(self.share_fit_dense(sequence_layer))
                else:
                    tmp.append(self.fit_denses[s_id](sequence_layer))
            sequence_output = tmp
            
        return sequence_output

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):

        _, pooled_output, sequence_output, att_output = self.bert(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
            output_hidden_states=True, output_attentions=True)
        
        sequence_output = self.do_fit_dense(sequence_output)
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits, att_output, sequence_output
    
    def freeze_shallow_layers(self):
        self.bert.freeze_shallow_layers()
    

class SkipBertForSequenceClassificationPrediction(SkipBertForSequenceClassification):
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        
        assert not self.training
        
        _, pooled_output, sequence_output, att_output = self.bert(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
            output_hidden_states=True, output_attentions=True)
        
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss = torch.tensor(0.)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )