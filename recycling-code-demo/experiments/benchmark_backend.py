import itertools
import os
import random
import shutil
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import NamedTuple, Type

import espresso_config as es
import numpy as np
import pandas as pd
import plotly.express as px
import torch
from tqdm import tqdm

from s2re.backend import BackendRegistry
from s2re.backend.base import BaseKVStorage
from s2re.utils import get_file_size


class Results(NamedTuple):
    num_sequences: int
    chunk_size: int
    sequence_length: int
    embedding_dim: int
    backend_name: str
    embedding_type: str
    seed: int
    write_time: float
    read_time: float
    db_size: int


def bench(
    num_sequences: int,
    chunk_size: int,
    sequence_length: int,
    embedding_dim: int,
    path: Path,
    Backend: Type[BaseKVStorage],
    embedding_lib: str,
    embedding_type: str,
    seed: int = 42,
) -> Results:
    random.seed(seed)
    np.random.seed(seed)

    if embedding_lib == "torch":
        dtype = getattr(torch, embedding_type)
        array = torch.randn(sequence_length, embedding_dim).type(dtype)
    else:
        dtype = getattr(np, embedding_type)
        array = np.random.randn(sequence_length, embedding_dim).astype(dtype)

    backend_name = Backend.__name__

    try:
        db = Backend(path)
        order = [str(i) for i in range(num_sequences)]
        start = time.time()

        for i in range(0, len(order), chunk_size):
            e = min(i + chunk_size, len(order))
            keys = order[i:e]
            db.batch_write(keys=keys, values=[array for _ in keys])
        write_time = time.time() - start

        db_size = sum(get_file_size(f) for f in db.files())

        # fresh new database
        del db
        db = Backend(path, read_only=True)

        # read in different order from read
        random.shuffle(order)

        start = time.time()
        for i in range(0, len(order), chunk_size):
            e = min(i + chunk_size, len(order))
            db.batch_read(order[i:e])

        read_time = time.time() - start
        del db

    finally:
        for f in Backend.files(path):
            if f.exists():
                os.remove(f)
        if path.exists() and path.is_dir():
            shutil.rmtree(path)

    return Results(
        num_sequences=num_sequences,
        chunk_size=chunk_size,
        sequence_length=sequence_length,
        embedding_dim=embedding_dim,
        backend_name=backend_name,
        embedding_type=embedding_type,
        seed=seed,
        write_time=write_time,
        read_time=read_time,
        # in MB
        db_size=db_size / 10e6,
    )


class Config(es.ConfigNode):
    mode: es.ConfigParam(str)

    num_sequences: es.ConfigParam(list) = [1000]
    sequence_length: es.ConfigParam(list) = [256]
    embedding_dim: es.ConfigParam(list) = [1024]
    chunk_size: es.ConfigParam(list) = [128]
    backend: es.ConfigParam(str) = "rocksdict"
    embedding_type: es.ConfigParam(str) = "float32"
    embedding_lib: es.ConfigParam(str) = "torch"
    path: es.ConfigParam(Path) = Path(tempfile.gettempdir()) / "bench.db"
    save_dir: es.ConfigParam(Path) = Path(__file__).parent / ".." / "results"


@es.cli(Config)
def main(config: Config):
    experiments = OrderedDict(
        num_sequences=config.num_sequences,
        sequence_length=config.sequence_length,
        embedding_dim=config.embedding_dim,
        chunk_size=config.chunk_size,
    )
    experiments_combinations = itertools.product(*experiments.values())
    if config.mode == "plot":
        experiments_combinations = tqdm(experiments_combinations, unit=" ex")

    results = []
    for n, s, d, c in experiments_combinations:
        result = bench(
            num_sequences=n,
            chunk_size=c,
            sequence_length=s,
            embedding_dim=d,
            embedding_lib=config.embedding_lib,
            embedding_type=config.embedding_type,
            path=config.path,
            Backend=BackendRegistry.get(config.backend),
        )
        results.append(result)
        if config.mode != "plot":
            print(result)

    if config.mode == "plot":
        if not config.save_dir.exists():
            config.save_dir.mkdir(parents=True)

        results_df = pd.DataFrame(results, columns=Results._fields)

        results_df.to_csv(config.save_dir / "test_backend.csv")

        fig = px.line(
            results_df,
            title="Write Time",
            x="num_sequences",
            y="write_time",
            color="chunk_size",
        )
        fig.write_image(config.save_dir / "write_time.jpg")

        fig = px.line(
            results_df,
            title="Read Time",
            x="num_sequences",
            y="read_time",
            color="chunk_size",
        )
        fig.write_image(config.save_dir / "read_time.jpg")

        fig = px.line(
            results_df, title="Storage", x="num_sequences", y="db_size"
        )
        fig.write_image(config.save_dir / "db_size.jpg")


if __name__ == "__main__":
    main()
