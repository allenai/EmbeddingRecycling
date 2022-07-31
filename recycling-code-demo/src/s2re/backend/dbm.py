import dbm
from pathlib import Path
from typing import Iterable, Literal, Sequence, Type

from ..types import HookComboKeyType, HookComboValueType
from .base import BackendRegistry, BaseKVStorage


@BackendRegistry.reg("dbm")
class DbmStorage(BaseKVStorage):
    db: dbm

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.access_type: Literal["r", "c"] = "r" if self.read_only else "c"
        self.str_path = str(self.path)

    @classmethod
    def files(cls: Type["DbmStorage"], path: Path) -> Iterable[Path]:
        path = Path(path)  # making sure it's a pathlib.Path
        return (f for f in path.parent.glob(f"{path.name}.*") if f.is_file())

    def batch_read(
        self, keys: Iterable[HookComboKeyType]
    ) -> Sequence[HookComboValueType]:
        with dbm.open(self.str_path, self.access_type) as db:
            return [self.sr.deserialize(db[self.sr.key(k)]) for k in keys]

    def batch_write(
        self,
        keys: Iterable[HookComboKeyType],
        values: Iterable[HookComboValueType],
    ) -> None:
        with dbm.open(self.str_path, self.access_type) as db:
            for key, array in zip(keys, values):
                db[self.sr.key(key)] = self.sr.serialize(array)

    def batch_delete(self, keys: Iterable[HookComboKeyType]) -> None:
        with dbm.open(self.str_path, self.access_type) as db:
            for key in keys:
                del db[self.sr.key(key)]
