from typing import Union

from torch.nn import Module, ModuleList
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import (
    BertEmbeddings,
    BertEncoder,
    BertForMaskedLM,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertLayer,
    BertModel,
)

from s2re import CachedLayer, CacheKeyLookup, NoOpWhenCached

__all__ = [
    "CachedBertForSequenceClassification",
    "CachedBertForMaskedLM",
    "CachedBertForTokenClassification",
]


class CachedBertEmbeddings(CacheKeyLookup, BertEmbeddings):
    def get_cache_arg_name_or_pos(self):
        return "input_ids", 0

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class NoOpWhenCachedBertLayer(NoOpWhenCached, BertLayer):
    def get_cache_hit_value(self):
        return [None]


class CachedBertLayer(CachedLayer, BertLayer):
    ...


class CachedBertConfig(BertConfig):
    def __init__(
        self: "CachedBertConfig",
        position_to_cache: Union[int, float] = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.position_to_cache = position_to_cache


class CachedBertEncoder(BertEncoder):
    def __init__(self, config: CachedBertConfig):
        super().__init__(config)

        if hasattr(config, "position_to_cache"):
            position_to_cache = int(getattr(config, "position_to_cache"))
        else:
            position_to_cache = max(round(config.num_hidden_layers / 2) - 1, 0)

        def layer_factory(layer_idx: int) -> Module:
            if layer_idx < position_to_cache:
                return NoOpWhenCachedBertLayer(config)
            if layer_idx == position_to_cache:
                return CachedBertLayer(config)
            if layer_idx > position_to_cache:
                return BertLayer(config)
            raise ValueError(f"Invalid layer index {layer_idx}.")

        self.layer = ModuleList(
            [layer_factory(i) for i in range(config.num_hidden_layers)]
        )


class CachedBertModel(BertModel):
    def __init__(
        self, config: CachedBertConfig, add_pooling_layer: bool = True
    ):
        super().__init__(config=config, add_pooling_layer=add_pooling_layer)
        self.embeddings = CachedBertEmbeddings(config)
        self.encoder = CachedBertEncoder(config)


class CachedBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config: CachedBertConfig) -> None:
        super().__init__(config)
        self.bert = CachedBertModel(config)


class CachedBertForTokenClassification(BertForTokenClassification):
    def __init__(self, config: CachedBertConfig) -> None:
        super().__init__(config)
        self.bert = CachedBertModel(config)


class CachedBertForMaskedLM(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.bert = CachedBertModel(config)
