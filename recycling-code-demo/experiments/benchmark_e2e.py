#! /usr/bin/env	python3

import contextlib
import json
import os
import shutil
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

import datasets
import springs as sp
import torch
import tqdm
import transformers
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification  # type: ignore
from transformers import BertTokenizer  # type: ignore
from transformers import DataCollatorWithPadding  # type: ignore
from transformers import PreTrainedModel  # type: ignore

from s2re import CachingHook
from s2re.backend import BackendRegistry
from s2re.models.bert import CachedBertForSequenceClassification
from s2re.utils import get_file_size


@sp.dataclass
class CacheConfig(sp.DataClass):
    _target_: str = sp.Target.to_string(CachingHook)
    path: str = "/tmp/r3"
    backend: str = "leveldb"


@sp.dataclass
class TokenizerConfig(sp.DataClass):
    _target_: str = sp.Target.to_string(BertTokenizer.from_pretrained)
    pretrained_model_name_or_path: str = "${backbone}"


@sp.dataclass
class ModelConfig(sp.DataClass):
    _target_: str = sp.Target.to_string(
        BertForSequenceClassification.from_pretrained
    )
    pretrained_model_name_or_path: str = "${backbone}"


@sp.dataclass
class LoaderConfig(sp.DataClass):
    _target_: str = sp.Target.to_string(datasets.load.load_dataset)
    path: str = "qasper"
    split: str = "train"


@sp.dataclass
class DatasetConfig(sp.DataClass):
    feature: str = "full_text.paragraphs"
    num_proc: int = 1
    num_samples: int = -1
    loader: LoaderConfig = LoaderConfig()


@sp.dataclass
class FetchConfig(sp.DataClass):
    fetch_ahead: int = 32
    fetch_spawn: str = "thread"
    fetch_timeout: float = 0.1
    fetch_retry_count: int = 10


@sp.dataclass
class Experiment(sp.DataClass):
    backbone: str = "bert-base-uncased"
    batch_size: int = 16
    device: str = "cpu"
    keep_cache: bool = False
    logs_path: Optional[str] = None
    steps: List[int] = sp.field(default_factory=lambda: [2, 3, 4, 5])
    is_train: bool = False

    cache: CacheConfig = CacheConfig()
    tokenizer: TokenizerConfig = TokenizerConfig()
    model: ModelConfig = ModelConfig()
    fetch: FetchConfig = FetchConfig()
    cached_model: ModelConfig = ModelConfig(
        _target_=sp.Target.to_string(
            CachedBertForSequenceClassification.from_pretrained
        )
    )
    dataset: DatasetConfig = DatasetConfig()


@contextlib.contextmanager
def maybe_grad(grad: bool) -> Iterator[None]:
    prev = torch.is_grad_enabled()
    try:
        torch.set_grad_enabled(grad)
        yield None
    finally:
        torch.set_grad_enabled(prev)


def run_model(
    model: transformers.modeling_utils.PreTrainedModel,
    data_loader: Union[DataLoader, Iterable[Any]],
    num_batches: Optional[int] = None,
    step: Optional[str] = None,
    is_train: bool = False,
    warmup: int = 2,
) -> float:

    model = model.train() if is_train else model.eval()

    if num_batches is None:
        if not isinstance(data_loader, DataLoader):
            raise ValueError(
                "num_batches must be specified if data_loader is "
                "not a DataLoader"
            )
        num_batches = len(data_loader)

    data_loader = (elem for elem in data_loader)

    if model.device == "cuda":

        def _sync():
            torch.cuda.synchronize()

    else:

        def _sync():
            ...

    delta = None

    if is_train:
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    else:
        optimizer = None

    labels = None
    with maybe_grad(grad=is_train):
        # Warmup
        for i, batch in enumerate(data_loader, start=1):
            if i == 0:
                # we sleep a bit to make sure the prefetching has loaded
                # some content
                time.sleep(3)

            batch = {k: v.to(model.device) for k, v in batch.items()}
            if labels is None and is_train:
                labels = torch.randint(
                    model.num_labels,  # type: ignore
                    (batch["input_ids"].size(0),),
                    device=model.device,
                )

            _ = model(**batch)

            _sync()
            num_batches -= 1
            if i >= warmup:
                break

        # Actual measuring
        cnt = 0
        progress_bar = tqdm.tqdm(
            data_loader, unit="ba", desc=step, total=num_batches
        )
        start = time.time()
        for batch in progress_bar:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            batch["labels"] = labels

            output = model(**batch)

            if optimizer:
                output.loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            cnt += 1
            _sync()

        delta = time.time() - start

    if delta is None:
        raise RuntimeError("No measurement occurred.")

    return delta / cnt


def _recursive_flatten(seq):
    for el in seq:
        if isinstance(el, (list, tuple)):
            yield from _recursive_flatten(el)
        else:
            yield el


def recursive_flatten(batch: Dict[str, list], column_name: str):
    return {column_name: list(_recursive_flatten(batch[column_name]))}


def tokenize(
    element: Dict[str, Any],
    column_name: str,
    tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase,
    **kw: Any
) -> transformers.tokenization_utils_base.BatchEncoding:
    return tokenizer(element[column_name], **kw)


def count_tokens(element: Dict[str, list], column_name: str):
    return {"count_tokens": len(element[column_name])}


def keep_n(element: Dict[str, Any], idx: int, n: int) -> bool:
    return idx < n


@sp.cli(Experiment)
def main(config: Experiment):
    tokenizer = sp.init(
        config.tokenizer,
        transformers.tokenization_utils_base.PreTrainedTokenizerBase,
    )

    dataset = sp.init(config.dataset.loader, datasets.arrow_dataset.Dataset)
    dataset = dataset.flatten()  # type: ignore
    num_papers = len(dataset) if config.dataset.num_samples else None

    # this is a bit convoluted, but: the backend_class is going
    # to be useful when we want to get the files in the cache,
    # but have closed the caching hook already.
    backend_cls = BackendRegistry.get(config.cache.backend)

    dataset = dataset.map(
        partial(recursive_flatten, column_name=config.dataset.feature),
        num_proc=config.dataset.num_proc,
        batched=True,
        remove_columns=dataset.column_names,
    )

    if config.dataset.num_samples > -1:
        dataset = dataset.filter(
            partial(keep_n, n=config.dataset.num_samples), with_indices=True
        )

    dataset = dataset.map(
        partial(
            tokenize,
            tokenizer=tokenizer,
            truncation=True,
            column_name=config.dataset.feature,
        ),
        remove_columns=dataset.column_names,
    )

    num_tokens = sum(
        dataset.map(partial(count_tokens, column_name="input_ids"))[
            "count_tokens"
        ]
    )
    num_sentences = len(dataset)

    dataset = dataset.with_format("torch")
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, return_tensors="pt"
    )
    data_loader = DataLoader(
        dataset,
        drop_last=True,
        batch_size=config.batch_size,
        collate_fn=data_collator,
    )

    if not config.keep_cache and os.path.exists(config.cache.path):
        shutil.rmtree(config.cache.path)

    if 1 in config.steps:
        # FIRST TEST: Vanilla transformer model as implemented in HuggingFace
        model = sp.init(
            config.model, transformers.modeling_utils.PreTrainedModel
        )
        model.to(config.device)
        vanilla_delta = run_model(
            model=model,
            data_loader=data_loader,
            step="vanilla",
            is_train=config.is_train,
        )
        del model
    else:
        vanilla_delta = None

    if 2 in config.steps:
        # SECOND TEST: Cached transformer model, but with caching disabled
        model = sp.init(
            config.cached_model, transformers.modeling_utils.PreTrainedModel
        )
        model.to(config.device)
        caching_off_delta = run_model(
            model=model,
            data_loader=data_loader,
            step="caching_off",
            is_train=config.is_train,
        )
        del model
    else:
        caching_off_delta = None

    if 3 in config.steps:
        # THIRD TEST: Cached transformer model, now saving the cache
        model = sp.init(
            config.cached_model, transformers.modeling_utils.PreTrainedModel
        )
        caching_hook = sp.init(config.cache, CachingHook)

        model.to(config.device)
        with caching_hook.record(model):
            caching_on_delta = run_model(
                model=model,
                data_loader=data_loader,
                step="caching_on",
                is_train=False,
            )
        del model, caching_hook
    else:
        caching_on_delta = None

    if 4 in config.steps:
        # we can't use caching_hook.storage because we need to delete
        # the caching_hook to close it!
        cache_size = sum(
            get_file_size(f) for f in backend_cls.files(config.cache.path)
        )

        # FOURTH TEST: Cached transformer model, using the cache
        model = sp.init(
            config.cached_model, transformers.modeling_utils.PreTrainedModel
        )
        caching_hook = sp.init(config.cache, CachingHook)
        model.to(config.device)
        caching_context_fn = (
            caching_hook.train if config.is_train else caching_hook.use
        )
        with caching_context_fn(model):
            caching_use_delta = run_model(
                model=model,
                data_loader=data_loader,
                step="caching_use",
                is_train=config.is_train,
            )
        del model, caching_hook
    else:
        caching_use_delta = None
        cache_size = None

    if 5 in config.steps:
        model = sp.init(config.cached_model, PreTrainedModel)
        caching_hook = sp.init(config.cache, CachingHook)
        model.to(config.device)
        caching_context_fn = (
            caching_hook.train if config.is_train else caching_hook.use
        )

        with caching_context_fn(
            model,
            fetch_key_fn=lambda x: x["input_ids"],
            **sp.to_dict(config.fetch),  # type: ignore
        ) as session:

            caching_prefetch_delta = run_model(
                model=model,
                data_loader=session.iterate(data_loader),
                num_batches=len(data_loader),
                step="caching_prefetch",
                is_train=config.is_train,
            )
        del model, caching_hook
    else:
        caching_prefetch_delta = None

    report = {
        "num_papers": num_papers,
        "num_tokens": num_tokens,
        "num_sentences": num_sentences,
        "vanilla": vanilla_delta,
        "caching_off": caching_off_delta,
        "caching_on": caching_on_delta,
        "caching_use": caching_use_delta,
        "caching_prefetch": caching_prefetch_delta,
        "cache_size": cache_size,
    }

    # print the report for ease of reading
    print(json.dumps(report, sort_keys=True, indent=2))

    # if a path to logs is provided, write the report and the experiment
    # configuration in jsonl format
    if config.logs_path:
        with open(config.logs_path, "a", encoding="utf-8") as f:
            log_entry = {
                "config": sp.to_dict(config),  # type: ignore
                "report": report,
            }
            f.write(json.dumps(log_entry, sort_keys=True) + "\n")

    if not config.keep_cache:
        # delete all cache files if required
        for fn in backend_cls.files(config.cache.path):
            os.remove(fn)
        if Path(config.cache.path).exists():
            shutil.rmtree(config.cache.path, ignore_errors=True)


if __name__ == "__main__":
    main()
