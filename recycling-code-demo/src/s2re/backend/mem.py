import pickle
from pathlib import Path
from typing import Iterable, Sequence, Type

from .base import BaseKVStorage, BackendRegistry
from ..types import HookComboKeyType, HookComboValueType


@BackendRegistry.reg('mem')
class MemoryStorage(BaseKVStorage):
    db: dict

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.db = {}

        if self.path.exists():
            with open(self.path, 'rb') as f:
                while True:
                    try:
                        key, value = pickle.load(f)
                        self.db[key] = value
                    except EOFError:
                        break

    @classmethod
    def files(cls: Type['MemoryStorage'], path: Path) -> Iterable[Path]:
        return [path]

    def batch_read(
        self,
        keys: Iterable[HookComboKeyType]
    ) -> Sequence[HookComboValueType]:
        # return [self.sr.deserialize(self.db[self.sr.key(k)]) for k in keys]
        return [self.db[key] for key in keys]

    def batch_write(
        self,
        keys: Iterable[HookComboKeyType],
        values: Iterable[HookComboValueType]
    ) -> None:
        for key, array in zip(keys, values):
            # key = self.sr.key(key)
            # array = self.sr.serialize(array)

            if key in self.db:
                raise KeyError(f'Key {key} already exists; cannot overwrite '
                               'with MemoryStorage backend')

            with open(self.path, 'ab') as f:
                pickle.dump((key, array), f)

            self.db[key] = array

    def batch_delete(self, keys: Iterable[HookComboKeyType]) -> None:
        raise NotImplementedError('MemoryStorage backend does not support'
                                  'deletion of specific keys')
