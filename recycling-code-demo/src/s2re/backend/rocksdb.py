import multiprocessing
from pathlib import Path
from typing import Iterable, Sequence, Type
try:
    import rocksdb
except ImportError:
    rocksdb = None

from .base import BaseKVStorage, BackendRegistry
from ..types import HookComboKeyType, HookComboValueType


@BackendRegistry.reg('rocksdb')
class RocksDBStorage(BaseKVStorage):
    db: rocksdb.DB if rocksdb else None

    def __init__(self,
                 *args,
                 compression: bool = True,
                 compression_type: str = 'snappy',
                 n_proc: int = None,
                 optimize_filters_for_hits: bool = True,
                 allow_mmap_reads: bool = True,
                 use_adaptive_mutex: bool = True,
                 **kwargs):
        super().__init__(*args, **kwargs)

        n_proc = n_proc or multiprocessing.cpu_count()
        options = rocksdb.Options(
            create_if_missing=True,
            compression_opts=dict(
                enabled=compression,
                compression_type=rocksdb.CompressionType(compression_type),
                parallel_threads=n_proc
            ),
            optimize_filters_for_hits=optimize_filters_for_hits,
            max_background_jobs=n_proc,
            allow_mmap_reads=allow_mmap_reads,
            use_adaptive_mutex=use_adaptive_mutex
        )
        options.IncreaseParallelism(n_proc)
        self.db = rocksdb.DB(str(self.path), options)

    @classmethod
    def files(cls: Type['RocksDBStorage'], path: Path) -> Iterable[Path]:
        path = Path(path)   # making sure it's a pathlib.Path
        return (f for f in Path(path).glob('**/*') if f.is_file())

    def batch_read(
        self: 'RocksDBStorage',
        keys: Iterable[HookComboKeyType]
    ) -> Sequence[HookComboValueType]:
        return [self.sr.deserialize(self.db.get(self.sr.key(k)))
                for k in keys]

    def batch_write(
        self: 'RocksDBStorage',
        keys: Iterable[HookComboKeyType],
        values: Iterable[HookComboValueType]
    ) -> None:
        wb = rocksdb.WriteBatch()
        for key, array in zip(keys, values):
            wb.put(self.sr.key(key), self.sr.serialize(array))
        self.db.write(wb)

    def batch_delete(self: 'RocksDBStorage',
                     keys: Iterable[HookComboKeyType]) -> None:
        wb = rocksdb.WriteBatch()
        for key in keys:
            wb.delete(self.sr.key(key))
        self.db.write(wb)
