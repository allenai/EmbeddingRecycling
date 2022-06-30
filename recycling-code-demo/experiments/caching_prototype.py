from abc import ABC
from collections import abc
from pathlib import Path
import transformers
import torch
from contextlib import contextmanager
from typing import Dict, Sequence, Union, Protocol, Optional
import numpy as np


SingleKeyType = torch.Tensor
KeyType = Union[SingleKeyType, Sequence[SingleKeyType]]
SingleValueType = Union[torch.Tensor,
                        Sequence[torch.Tensor],
                        Dict[str, torch.Tensor]]
ValueType = Union[SingleValueType, Sequence[SingleValueType]]


class StoreFnProtocol(Protocol):
    """Protocol for the function that caches a tensor"""
    def __call__(self, key: KeyType, value: ValueType): ...


class FetchFnProtocol(Protocol):
    """Protocol for the function that retrieves a cached tensor"""
    def __call__(self, key: KeyType, device: str): ...


class CachingSession:
    """An object that contains tools to work in a caching session:
    the function to store output after computation, or the function
    to retrieve the cached output.

    Calling the `recording` attribute tells which mode this caching
    session is running: if True, it is set to record partial
    computations; if False, partial computations are retrieved instead
    of re-running layers.
    """
    def __init__(self,
                 store_fn: Optional[StoreFnProtocol] = None,
                 fetch_fn: Optional[FetchFnProtocol] = None):
        self._key = None
        self.store_fn = store_fn
        self.fetch_fn = fetch_fn

    @property
    def recording(self):
        return self.store_fn is not None

    def store(self, value: torch.Tensor):
        if self._key is None:
            raise RuntimeError('Key not provided')

        if self.store_fn is None:
            raise RuntimeError('Not set for recording!')

        self.store_fn(key=self._key, value=value)
        self._key = None

    def fetch(self) -> torch.Tensor:
        if self._key is None:
            raise RuntimeError('Key not provided')

        if self.fetch_fn is None:
            raise RuntimeError('Not set for fetching!')

        out = self.fetch_fn(self._key)
        self._key = None
        return out

    def key(self, key: torch.TensorType):
        if self._key is not None:
            raise RuntimeError('Key is already set')
        self._key = key


class BaseModuleWithCaching(ABC):
    @property
    def cache(self) -> Union[CachingSession, None]:
        return getattr(self, '__caching_session__', None)

    def set_cache(self, cache_session: CachingSession):
        return setattr(self, '__caching_session__', cache_session)

    def unset_cache(self):
        return delattr(self, '__caching_session__')

    def get_cache_hit_value(self):
        """Value to return when there is a cache hit.
        Subclasses might customize it for compatibility reasons."""
        return None


class BaseMixinWithBaseModuleWithCaching(BaseModuleWithCaching,
                                         torch.nn.Module):
    """Mixin for type annotations"""
    ...


class CachingHook:
    def __init__(self, path: Union[str, Path] = None):
        self.path = path
        self.storage = {}

    def _cast_key(self, t: torch.tensor) -> bytes:
        return self._cast_value(t).numpy().tobytes()

    def _cast_value(self, t: torch.tensor) -> np.ndarray:
        return t.detach().cpu()

    def store(self, key: KeyType, value: ValueType):
        if isinstance(key, torch.Tensor):
            key = [key]
            value = [value]

        for single_key, single_value in zip(key, value):
            if isinstance(single_value, abc.Mapping):
                single_value = {k: self._cast_value(v)
                                for k, v in single_value.items()}
            elif isinstance(single_value, abc.Sequence):
                single_value = [self._cast_value(v) for v in single_value]
            else:
                single_value = self._cast_value(single_value)

            self.storage[self._cast_key(single_key)] = single_value

    def fetch(self, key: KeyType, device: str = 'cpu') -> ValueType:
        if isinstance(key, torch.Tensor):
            single_element = True
            key = [key]
        else:
            single_element = False

        values = []
        for single_key in key:
            single_value = self.storage.get(self._cast_key(single_key))
            if isinstance(single_value, abc.Mapping):
                single_value = {k: v.device(device)
                                for k, v in single_value.items()}
            elif isinstance(single_value, abc.Sequence):
                single_value = [v.device(device) for v in single_value]
            values.append(single_value)

        if single_element:
            return values[0]

    def find_all_caching_modules(
        self,
        module: torch.nn.Module
    ) -> Sequence['BaseMixinWithBaseModuleWithCaching']:
        out = [m for _, m in module.named_modules()
               if isinstance(m, BaseModuleWithCaching)]
        if isinstance(module, BaseModuleWithCaching):
            out.insert(0, module)
        return out

    @contextmanager
    def record(self, module: torch.nn.Module) -> CachingSession:
        caching_modules = self.find_all_caching_modules(module)
        session = CachingSession(store_fn=self.store)
        try:
            [m.set_cache(session) for m in caching_modules]
            yield session
        finally:
            [m.unset_cache() for m in caching_modules]

    @contextmanager
    def use(self, module: torch.nn.Module) -> CachingSession:
        caching_modules = self.find_all_caching_modules(module)
        session = CachingSession(fetch_fn=self.fetch)
        try:
            [m.set_cache(session) for m in caching_modules]
            yield session
        finally:
            [m.unset_cache() for m in caching_modules]


class CacheKeyLookup(BaseModuleWithCaching):
    def forward(self, input_ids: torch.LongTensor, *args, **kwargs):
        if self.cache is None:
            return super().forward(*args, input_ids=input_ids, **kwargs)

        value = (super().forward(*args, input_ids=input_ids, **kwargs)
                 if self.cache.recording else self.get_cache_hit_value)

        self.cache.key(input_ids)
        return value


class NoOpWhenCached(BaseModuleWithCaching):
    def forward(self, *args, **kwargs):
        if self.cache is None or self.cache.recording:
            return super().forward(*args, **kwargs)
        else:
            return self.get_cache_hit_value


class CachedLayer(BaseModuleWithCaching):
    def forward(self, *args, **kwargs):
        if self.cache is None:
            return super().forward(*args, **kwargs)
        elif self.cache.recording:
            out = super().forward(*args, **kwargs)
            self.cache.store(out)
        else:
            out = self.cache.fetch()
        return out


##############


class CachedBertEmbeddings(
    CacheKeyLookup,
    transformers.models.bert.modeling_bert.BertEmbeddings
):
    ...


class NoOpWhenCachedBertLayer(
    NoOpWhenCached,
    transformers.models.bert.modeling_bert.BertLayer
):
    @property
    def get_cache_hit_value(self):
        return [None]


class CachedBertLayer(
    CachedLayer,
    transformers.models.bert.modeling_bert.BertLayer
):
    ...


class CachedBertEncoder(transformers.models.bert.modeling_bert.BertEncoder):
    def __init__(self, config):
        super().__init__(config)

        pos_to_cache = getattr(
            config,
            'position_to_cache',
            max(round(config.num_hidden_layers / 2) - 1, 0)
        )
        self.layer = torch.nn.ModuleList([
            NoOpWhenCachedBertLayer(config)
            if i < pos_to_cache else (
                CachedBertLayer(config)
                if i == pos_to_cache else
                transformers.models.bert.modeling_bert.BertLayer(config)
            ) for i in range(config.num_hidden_layers)
        ])


class BertModel(transformers.BertModel):
    def __init__(self,
                 config: transformers.BertConfig,
                 add_pooling_layer: bool = True):
        super().__init__(config=config, add_pooling_layer=add_pooling_layer)
        self.embeddings = CachedBertEmbeddings(config)
        self.encoder = CachedBertEncoder(config)


class BertForSequenceClassification(
    transformers.BertForSequenceClassification
):
    def __init__(self,  config: transformers.BertConfig) -> None:
        super().__init__(config)
        self.bert = BertModel(config)


def main():
    bb = 'nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large'
    tokenizer = transformers.BertTokenizer.from_pretrained(bb)
    model = BertForSequenceClassification.from_pretrained(bb)

    text = 'I love apple pie'
    model_input = tokenizer(text, return_tensors='pt')
    output = model(**model_input)
    print('No caching: ', output)

    caching_hook = CachingHook()
    with caching_hook.record(model):
        output = model(**model_input)
        print('Recoding: ', output)

    with caching_hook.use(model):
        output = model(**model_input)
        print('Using: ', output)



if __name__ == '__main__':
    main()
