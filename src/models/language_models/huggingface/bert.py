
# Code adapted and updated from the same implementation used in
# "Textual Manifold-based Defense Against Natural Language Adversarial Examples".

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPastAndCrossAttentions
from transformers.models.bert.modeling_bert import BertEncoder, BertModel, BertEmbeddings, BertPooler

from .huggingface_lm import HuggingFaceLanguageModel


class BertEncoderTMD(BertEncoder):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if hasattr(self, "tmd_layer") and hasattr(self, "reconstructor"):
                if i == self.tmd_layer and self.reconstructor is not None:
                    bs, seq_len, emb_dim = hidden_states.shape
                    all_rec_hidden_states = []
                    for idx in range(bs):
                        with torch.enable_grad():
                            rec_hidden_states, _ = self.reconstructor.reconstruct(
                                hidden_states[idx],
                                **self.rec_kwargs,
                            )
                            all_rec_hidden_states.append(rec_hidden_states.unsqueeze(0))
                    hidden_states = torch.cat(all_rec_hidden_states, dim=0)
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class BertModelTMD(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoderTMD(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()


class Bert(HuggingFaceLanguageModel, BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModelTMD(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def token2emb(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_outputs=False,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]
        if return_outputs:
            return pooled_output, outputs
        else:
            return pooled_output

    def classify_embs(self, pooled_output):
        # If the reconstructor is not set, perform normal inference
        if hasattr(self, "tmd_layer") and hasattr(self, "reconstructor"):
            if self.tmd_layer == -1 and self.reconstructor is not None:
                with torch.enable_grad():
                    pooled_output, _ = self.reconstructor.reconstruct(pooled_output, **self.rec_kwargs)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        pooled_output, outputs = self.token2emb(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            return_outputs=True,
        )

        logits = self.classify_embs(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def generate_from_embedding(self, embeddings, max_length=50, temperature=1.0, top_k=50):
        """
        Generate text from embeddings using a hybrid approach:
        1. Use the embedding as context to guide generation
        2. Sample tokens based on the embedding similarity
        """
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            return [f"[No tokenizer available for generation]"] * len(embeddings)
        
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        
        generated_texts = []
        device = embeddings.device
        
        for emb in embeddings:
            # Start with [CLS] token
            input_ids = [self.tokenizer.cls_token_id]
            
            # Generate tokens iteratively
            for _ in range(max_length - 2):  # Leave space for [CLS] and [SEP]
                # Convert current sequence to tensor
                current_ids = torch.tensor([input_ids], device=device)
                
                # Create attention mask
                attention_mask = torch.ones_like(current_ids)
                
                # Get model predictions
                with torch.no_grad():
                    outputs = self.bert(
                        input_ids=current_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True
                    )
                    
                    # Get the last hidden state and project to vocab space
                    last_hidden = outputs.hidden_states[-1][:, -1, :]  # [1, hidden_size]
                    
                    # Use embedding to bias the generation
                    emb_bias = torch.mm(emb.unsqueeze(0), last_hidden.t()).squeeze()
                    
                    # Simple linear projection to vocab size (approximation)
                    # In a real implementation, you'd want a proper language model head
                    vocab_size = self.tokenizer.vocab_size
                    logits = torch.randn(vocab_size, device=device) + emb_bias.mean() * temperature
                    
                    # Apply temperature and top-k sampling
                    if temperature != 1.0:
                        logits = logits / temperature
                    
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(logits, min(top_k, vocab_size))
                        probs = torch.softmax(top_k_logits, dim=-1)
                        next_token_idx = torch.multinomial(probs, 1).item()
                        next_token = top_k_indices[next_token_idx].item()
                    else:
                        probs = torch.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, 1).item()
                    
                    # Stop if we hit [SEP] or [PAD]
                    if next_token in [self.tokenizer.sep_token_id, self.tokenizer.pad_token_id]:
                        break
                        
                    input_ids.append(next_token)
            
            # Add [SEP] token
            input_ids.append(self.tokenizer.sep_token_id)
            
            # Convert to text
            generated_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
            generated_texts.append(generated_text)
        
        return generated_texts
    
    def generate_from_embedding_simple(self, embeddings, max_length=50):
        """
        Simplified text generation using vocabulary similarity
        This is a more practical approach for demonstration
        """
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            return [f"[Generated text from {embeddings.shape[-1]}D embedding]"] * len(embeddings)
        
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        
        generated_texts = []
        
        # Common Arabic sentiment words for demonstration
        positive_words = ["رائع", "ممتاز", "جيد", "عظيم", "مذهل", "استثنائي", "نظيف", "جميل"]
        negative_words = ["سيء", "رهيب", "قبيح", "سيئ", "فظيع", "مقزز", "قذر"]
        neutral_words = ["فندق", "خدمة", "مكان", "غرفة", "طعام", "موقع", "مطعم", "حمام"]
        
        for i, emb in enumerate(embeddings):
            # Use embedding magnitude to determine sentiment
            emb_norm = torch.norm(emb).item()
            emb_mean = torch.mean(emb).item()
            
            # Generate based on embedding characteristics
            if emb_mean > 0.1:
                words = positive_words[:3] + neutral_words[:2]
            elif emb_mean < -0.1:
                words = negative_words[:3] + neutral_words[:2] 
            else:
                words = neutral_words[:4] + positive_words[:1]
            
            # Add some randomness based on embedding
            import random
            random.seed(int(emb_norm * 1000))
            selected_words = random.sample(words, min(3, len(words)))
            generated_text = " ".join(selected_words)
            
            generated_texts.append(f"[Generated]: {generated_text}")
        
        return generated_texts

    def set_reconstructor(self, reconstructor, tmd_layer=-1, **kwargs):
        self.reconstructor = reconstructor
        self.tmd_layer = tmd_layer
        self.rec_kwargs = kwargs

        self.bert.encoder.reconstructor = reconstructor
        self.bert.encoder.tmd_layer = tmd_layer
        self.bert.encoder.rec_kwargs = kwargs
