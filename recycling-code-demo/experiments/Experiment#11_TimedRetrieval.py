#! /usr/bin/env	python3

import os
import shutil
import time
from functools import partial
from pathlib import Path
from random import randrange
from typing import Any, Dict

import datasets
import springs as sp
import torch
import torch.nn as nn
import tqdm
import transformers
from torch.optim import Adam
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    DataCollatorWithPadding,
    get_scheduler,
)

from s2re import CachingHook
from s2re.backend import BackendRegistry
from s2re.models.bert import CachedBertForSequenceClassification
from s2re.utils import get_file_size


@sp.dataclass
class CacheConfig:
    _target_: str = sp.Target.to_string(CachingHook)
    path: str = "/tmp/s2re"
    backend: str = "leveldb"


@sp.dataclass
class TokenizerConfig:
    _target_: str = sp.Target.to_string(BertTokenizer.from_pretrained)
    pretrained_model_name_or_path: str = "${backbone}"


@sp.dataclass
class ModelConfig:
    _target_: str = sp.Target.to_string(
        BertForSequenceClassification.from_pretrained
    )
    pretrained_model_name_or_path: str = "${backbone}"


@sp.dataclass
class LoaderConfig:
    _target_: str = sp.Target.to_string(datasets.load_dataset)
    path: str = "qasper"
    split: str = "train"


@sp.dataclass
class DatasetConfig:
    feature: str = "full_text.paragraphs"
    num_proc: int = 1
    num_samples: int = -1
    loader: LoaderConfig = LoaderConfig()


@sp.dataclass
class Experiment:
    backbone: str = "bert-base-uncased"
    batch_size: int = 16
    device: str = "cpu"
    keep_cache: bool = False

    cache: CacheConfig = CacheConfig()
    tokenizer: TokenizerConfig = TokenizerConfig()
    model: ModelConfig = ModelConfig()
    cached_model: ModelConfig = ModelConfig(
        _target_=sp.Target.to_string(
            CachedBertForSequenceClassification.from_pretrained
        )
    )
    dataset: DatasetConfig = DatasetConfig()


def run_model(
    model: transformers.PreTrainedModel,
    data_loader: torch.utils.data.DataLoader,
    step: str = None,
) -> int:

    optimizer = Adam(model.parameters(), lr=1e-5)

    num_epochs = 1
    num_training_steps = num_epochs * len(data_loader)

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=num_training_steps,
    )

    if model.device == "cuda":
        _sync = lambda: torch.cuda.synchronize()  # noqa: E731
    else:
        _sync = lambda: None  # noqa: E731

    delta = None

    start = time.time()
    for batch in tqdm.tqdm(data_loader, unit="ba", desc=step):

        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)

        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        _sync()

    delta = time.time() - start

    if delta is None:
        raise RuntimeError("No measurement occurred.")

    return delta


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
    tokenizer: transformers.PreTrainedTokenizer,
    **kw: Dict[str, Any],
):

    tokenized_output = tokenizer(element[column_name], **kw)

    labels = [0]
    tokenized_output.update({"labels": labels})

    return tokenized_output


def count_tokens(element: Dict[str, list], column_name: str):
    return {"count_tokens": len(element[column_name])}


def keep_n(element: Dict[str, Any], idx: int, n: int) -> bool:
    return idx < n


################################################################
@sp.cli(Experiment)
def training_main(config: Experiment):

    tokenizer = sp.init(config.tokenizer)

    dataset = sp.init(config.dataset.loader)
    dataset = dataset.flatten()
    num_papers = len(dataset)

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
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, collate_fn=data_collator
    )

    # FIRST TEST: Vanilla transformer model as implemented in HuggingFace
    print("-----------------------------------------------------")
    print("First Test")
    model = sp.init(config.model)
    model.to(config.device)
    model.train()
    vanilla_delta = run_model(
        model=model, data_loader=data_loader, step="vanilla"
    )
    del model
    print("-----------------------------------------------------")

    print("-----------------------------------------------------")
    print("Second Test")
    # SECOND TEST: Cached transformer model, but with caching disabled
    model = sp.init(config.cached_model)
    model.to(config.device)
    model.train()
    caching_off_delta = run_model(
        model=model, data_loader=data_loader, step="caching_off"
    )
    del model
    print("-----------------------------------------------------")

    print("-----------------------------------------------------")
    print("Third Test")
    # THIRD TEST: Cached transformer model, now saving the cache
    model = sp.init(config.cached_model)
    caching_hook: CachingHook = sp.init(config.cache)
    model.to(config.device)
    model.train()
    with caching_hook.record(model):
        caching_on_delta = run_model(
            model=model, data_loader=data_loader, step="caching_on"
        )
    del model, caching_hook
    print("-----------------------------------------------------")

    # we can't use caching_hook.storage because we need to delete
    # the caching_hook to close it!
    cache_size = sum(
        get_file_size(f) for f in backend_cls.files(config.cache.path)
    )

    print("-----------------------------------------------------")
    print("Fourth Test")
    # FOURTH TEST: Cached transformer model, using the cache
    model = sp.init(config.cached_model)
    caching_hook: CachingHook = sp.init(config.cache)
    model.to(config.device)
    model.train()
    with caching_hook.train(model, backend="leveldb", path="/tmp/s2re"):
        caching_use_delta = run_model(
            model=model, data_loader=data_loader, step="caching_use"
        )
    del model, caching_hook
    print("-----------------------------------------------------")

    if config.dataset.num_samples < 0:
        # don't print if you have limits
        print(f"num_papers: {num_papers:,}")

    print(f"num_tokens: {num_tokens:,}")
    print(f"num_sentences: {num_sentences:,}")
    print(f"vanilla model: {vanilla_delta:.2f} s")
    print(f"caching_off model: {caching_off_delta:.2f} s")
    print(f"caching_on model: {caching_on_delta:.2f} s")
    print(f"caching_use model: {caching_use_delta:.2f} s")
    print(f"cache_size {cache_size / 1024**2:.2f} MB")

    if not config.keep_cache:

        # DELETE ALL CACHE FILES
        for fn in backend_cls.files(config.cache.path):
            os.remove(fn)
        if Path(config.cache.path).exists():
            shutil.rmtree(config.cache.path, ignore_errors=True)


################################################################

if __name__ == "__main__":
    training_main()
