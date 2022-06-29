from pathlib import Path
from typing import Iterable, Sequence, Type

from .base import BaseKVStorage, BackendRegistry
from ..types import HookComboKeyType, HookComboValueType

try:
    from unqlite import UnQLite
except ImportError:
    UnQLite = None


@BackendRegistry.reg('unqlite')
class UnQLiteStorage(BaseKVStorage):
    """An uqlite-backed key-value storage for numpy arrays."""
    db: UnQLite

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if UnQLite is None:
            raise ImportError('unqlite is not installed.')

        # Flags values are defined at https://github.com/coleifer/
        # unqlite-python/blob/06bbd668382080bf929cdef5247423ff4c0e5b32
        # /unqlite.pyx#L237-L246
        flags = 0x00000002 if self.read_only else 0x00000004
        self.db = UnQLite(filename=str(self.path), flags=flags)

    @classmethod
    def files(cls: Type['UnQLiteStorage'], path: Path) -> Iterable[Path]:
        return (Path(path), )

    def batch_read(
        self,
        keys: Iterable[HookComboKeyType]
    ) -> Sequence[HookComboValueType]:
        with self.db.transaction():
            arrays = [self.sr.deserialize(self.db.fetch(self.sr.key(k)))
                      for k in keys]
        return arrays

    def batch_write(
        self,
        keys: Iterable[HookComboKeyType],
        values: Iterable[HookComboValueType]
    ) -> None:
        with self.db.transaction():
            for key, array in zip(keys, values):
                self.db.store(self.sr.key(key), self.sr.serialize(array))
        self.db.commit()

    def batch_delete(self, keys: Iterable[HookComboKeyType]) -> None:
        with self.db.transaction():
            for key in keys:
                self.db.delete(self.sr.key(key))