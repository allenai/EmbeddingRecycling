# Embedding Recycling Demo

This is a sample implementation of embedding recycling.

To recreate the benchmark results reported in our manuscript, run:

```bash
cd /path/to/this/this/demo/directory
pip install -e .                 # install the embedding recycling demo code
pip install -r experiments.txt   # install the packages required for experiments

bash experiments/benchmark_e2e.sh
```

## How Recycling Works

To enable recycling of embeddings, one needs to (a) modify a torch model to indicate which layers to recycle, and which inputs to use as key for the cache, and (b) use a `s2re.CachingHook` to start a recycling session.
For a full example of embedding recycling, it is recommend to check the [benchmark_e2e.py](/experiments/benchmark_e2e.py) experiment.

### How to Modify a Model to Enable Recycling

[`s2re.models.bert`](/src/s2re/models/bert.py) contains an example of a BERT model for sequence classification that has been modified to support recycling. Essentially, modification consist modifying the appropriate layers to:

1. determine which input we want to use as key for the cache,
2. indicate which layers to skip during recycling, and
3. insert the recycled embeddings during the forward pass.

In case of a BERT model, we achieve (1) by inheriting from `CacheKeyLookup` and providing a `get_cache_arg_name_or_pos` function. Note that **inheritance order** matters here:  `CacheKeyLookup` must be the first superclass must precede the embedding layers we are trying to decorate.

```python
from s2re import CachedLayer, CacheKeyLookup, NoOpWhenCached

class CachedBertEmbeddings(CacheKeyLookup, BertEmbeddings):
    def get_cache_arg_name_or_pos(self):
        return 'input_ids', 0

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
```

For (2), we skip encoding layers by decorating them with `NoOpWhenCached`. Note that `get_cache_hit_value` returns a list here; this is to mimic the behavior of BertLayer, which returns a list of tensors during its forward pass.

```python
class NoOpWhenCachedBertLayer(NoOpWhenCached, BertLayer):
    def get_cache_hit_value(self):
        return [None]
```

Finally, for (3), we create a layer that uses the recycled embeddings when recycling is on. Note how this layer does not modify any of the functions in BertLayer.

```python
class CachedBertLayer(CachedLayer, BertLayer):
    ...
```

Finally, we combine all the new layers into a BERT model. This is easily accomplished by mimicking the original BERT code. Note how we recycling extactly half of the hidden layers.


```python
class CachedBertEncoder(BertEncoder):
    def __init__(self, config: BertConfig):
        super().__init__(config)

        position_to_cache = config.num_hidden_layers // 2

        self.layer = ModuleList(
            [NoOpWhenCachedBertLayer(config) if i < position_to_cache else (
                CachedBertLayer(config)  if i == position_to_cache
                else BertLayer(config))
            )
             for i in range(config.num_hidden_layers)]
        )


class CachedBertModel(BertModel):
    def __init__(self,
                 config: CachedBertConfig,
                 add_pooling_layer: bool = True):
        super().__init__(config=config, add_pooling_layer=add_pooling_layer)
        self.embeddings = CachedBertEmbeddings(config)
        self.encoder = CachedBertEncoder(config)
```

### How to Use Recycling

Once a cached model is create, we can simply use it by taking advantage of `CachingHook`. If we don't want to recycle embeddings, we can use the model as usual:

```python
model = CachedBertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text = ['I like apples', 'They make great apple pie']

batch = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
output = model(**batch)
```

To record embeddings to recycle, we use `CachingHook.record`:

```python
from s2re import CachingHook

hook = CachingHook(path='/tmp/r3', backend='leveldb')
with hook.record(model):
    output = model(**batch)
    ...
```

To use the embedding for training, use `CachingHook.train`:

```python
with hook.train(model):
    output = model(**batch)
    output.backward()
    ...
```

To use the embeddings for inference, use `CachingHook.use`:

```python
with hook.use(model):
    output = model(**batch)
    print(output.logits)
    ...
```
