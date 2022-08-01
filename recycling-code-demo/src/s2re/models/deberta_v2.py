from typing import Union

from torch.nn import Module, ModuleList
from transformers.models.deberta_v2.configuration_deberta_v2 import (
    DebertaV2Config
)
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2Embeddings,
    DebertaV2Encoder,
    DebertaV2ForMaskedLM,
    DebertaV2ForSequenceClassification,
    DebertaV2ForTokenClassification,
    DebertaV2Layer,
    DebertaV2Model,
)

from s2re import (
    CachedLayer, CacheKeyLookup, NoOpWhenCached, BaseModuleWithCaching
)

__all__ = [
    "CachedDebertaV2ForSequenceClassification",
    "CachedDebertaV2ForMaskedLM",
    "CachedDebertaV2ForTokenClassification",
]


class CachedDebertaV2Embeddings(CacheKeyLookup, DebertaV2Embeddings):
    def get_cache_arg_name_or_pos(self):
        return "input_ids", 0

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class NoOpWhenCachedDebertaV2Layer(NoOpWhenCached, DebertaV2Layer):
    def get_cache_hit_value(self):
        return [None]


class CachedDebertaV2Layer(CachedLayer, DebertaV2Layer):
    ...


class CachedDebertaV2Config(DebertaV2Config):
    def __init__(
        self: "CachedDebertaV2Config",
        position_to_cache: Union[int, float] = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.position_to_cache = position_to_cache


class CachedDebertaV2Encoder(BaseModuleWithCaching, DebertaV2Encoder):
    def __init__(self, config: CachedDebertaV2Config):
        super().__init__(config)

        if hasattr(config, "position_to_cache"):
            position_to_cache = int(getattr(config, "position_to_cache"))
        else:
            position_to_cache = max(
                round(config.num_hidden_layers / 2) - 1, 0
            )

        def layer_factory(layer_idx: int) -> Module:
            if layer_idx < position_to_cache:
                return NoOpWhenCachedDebertaV2Layer(config)
            if layer_idx == position_to_cache:
                return CachedDebertaV2Layer(config)
            if layer_idx > position_to_cache:
                return DebertaV2Layer(config)
            raise ValueError(f"Invalid layer index {layer_idx}.")

        self.layer = ModuleList(
            [layer_factory(i) for i in range(config.num_hidden_layers)]
        )

    def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
        if self.cache is None or self.cache.recording or self.cache.training:
            return super().get_rel_pos(
                hidden_states, query_states, relative_pos
            )
        else:
            return super().get_rel_pos(self.cache._key.unsqueeze(-1))


class CachedDebertaV2Model(DebertaV2Model):
    def __init__(
        self, config: CachedDebertaV2Config
    ):
        super().__init__(config=config)
        self.embeddings = CachedDebertaV2Embeddings(config)
        self.encoder = CachedDebertaV2Encoder(config)


class CachedDebertaV2ForSequenceClassification(DebertaV2ForSequenceClassification):
    def __init__(self, config: CachedDebertaV2Config) -> None:
        super().__init__(config)
        self.deberta = CachedDebertaV2Model(config)


class CachedDebertaV2ForTokenClassification(DebertaV2ForTokenClassification):
    def __init__(self, config: CachedDebertaV2Config) -> None:
        super().__init__(config)
        self.deberta = CachedDebertaV2Model(config)


class CachedDebertaV2ForMaskedLM(DebertaV2ForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.deberta = CachedDebertaV2Model(config)
