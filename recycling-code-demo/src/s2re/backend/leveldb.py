from pathlib import Path
from typing import Iterable, Sequence, Type

try:
    import plyvel
except ImportError:
    plyvel = None

from .base import BaseKVStorage, BackendRegistry
from ..types import HookComboKeyType, HookComboValueType


@BackendRegistry.reg('leveldb')
class LevelDBStorage(BaseKVStorage):
    db: plyvel.DB if plyvel else None

    def __init__(self,
                 *args,
                 compression: bool = True,
                 max_open_files: int = 1024,
                 max_file_size: int = 2 * 1024 * 1024,
                 write_buffer_size: int = 4 * 1024 * 1024,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.access_type = 'r' if self.read_only else 'c'

        self.db = plyvel.DB(name=str(self.path),
                            max_open_files=max_open_files,
                            create_if_missing=True,
                            max_file_size=max_file_size,
                            write_buffer_size=write_buffer_size,
                            compression='snappy' if compression else None)

    @classmethod
    def files(cls: Type['LevelDBStorage'], path: Path) -> Iterable[Path]:
        path = Path(path)   # making sure it's a pathlib.Path
        return (f for f in Path(path).glob('**/*') if f.is_file())

    def batch_read(
        self: 'LevelDBStorage',
        keys: Iterable[HookComboKeyType]
    ) -> Sequence[HookComboValueType]:
        return [self.sr.deserialize(self.db.get(self.sr.key(k)))
                for k in keys]

    def batch_write(
        self: 'LevelDBStorage',
        keys: Iterable[HookComboKeyType],
        values: Iterable[HookComboValueType]
    ) -> None:
        with self.db.write_batch() as wb:
            for key, array in zip(keys, values):
                wb.put(self.sr.key(key), self.sr.serialize(array))

    def batch_delete(self: 'LevelDBStorage',
                     keys: Iterable[HookComboKeyType]) -> None:
        with self.db.write_batch() as wb:
            for key in keys:
                wb.delete(self.sr.key(key))
