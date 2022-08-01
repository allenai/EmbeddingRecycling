import sys
import time
from pathlib import Path
from typing import Union

from torch.nn import ModuleList
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
)
from transformers.models.bert.modeling_bert import (
    BertEmbeddings,
    BertEncoder,
    BertLayer,
    BertModel,
)

try:
    from s2re import CachedLayer, CacheKeyLookup, CachingHook, NoOpWhenCached
except ImportError:
    src = Path(__file__).parent / ".." / "src"
    sys.path.append(str(src))
    from s2re import CachedLayer, CacheKeyLookup, CachingHook, NoOpWhenCached


class CachedBertEmbeddings(CacheKeyLookup, BertEmbeddings):
    ...


class NoOpWhenCachedBertLayer(NoOpWhenCached, BertLayer):
    @property
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

        position_to_cache = getattr(config, "position_to_cache", None)
        if position_to_cache is None:
            position_to_cache = max(
                round(config.num_hidden_layers / 2) - 1, 0
            )

        def layer_factory(layer_idx: int):
            if layer_idx < position_to_cache:
                return NoOpWhenCachedBertLayer(config)
            if layer_idx == position_to_cache:
                return CachedBertLayer(config)
            if layer_idx > position_to_cache:
                return BertLayer(config)

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


class BertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config: CachedBertConfig) -> None:
        super().__init__(config)
        self.bert = CachedBertModel(config)


def main():
    bb = "nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large"
    tokenizer = BertTokenizer.from_pretrained(bb)
    model = BertForSequenceClassification.from_pretrained(bb)

    text = "I love apple pie! " * 30  # "* 30" to make it longer
    N = 100

    model_input = tokenizer(text, return_tensors="pt")
    _ = model(**model_input)

    start = time.time()
    for _ in range(N):
        _ = model(**model_input)
    delta = (time.time() - start) / N
    print(f"No caching in {delta * 1e3:.3f} ms")

    caching_hook = CachingHook(backend="rocksdict", path="/tmp/rocksdict")
    with caching_hook.record(model):
        delta = 0
        for _ in range(N):
            start = time.time()
            _ = model(**model_input)
            delta += time.time() - start
            caching_hook.delete(model_input.input_ids)
        model(**model_input)
        print(f"Recording in {delta * 1e3:.3f} ms")

    with caching_hook.use(model):
        start = time.time()
        for _ in range(N):
            _ = model(**model_input)
        delta = (time.time() - start) / N
        print(f"Using in {delta * 1e3:.3f} ms")


if __name__ == "__main__":
    main()
