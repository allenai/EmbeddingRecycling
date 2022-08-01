from pathlib import Path
from typing import Iterable, Sequence, Type

from ..types import HookComboKeyType, HookComboValueType
from .base import BackendRegistry, BaseKVStorage

try:
    from rocksdict import AccessType, Options, Rdict, WriteBatch, WriteOptions
except ImportError:
    Rdict = None


@BackendRegistry.reg("rocksdict")
class RocksDictStorage(BaseKVStorage):
    db: Rdict

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if Rdict is None:
            raise ImportError("rocksdict is not installed.")

        access_type = (
            AccessType.read_only()
            if self.read_only
            else AccessType.read_write()
        )
        self.db = Rdict(
            path=str(self.path),
            options=Options(raw_mode=True),
            access_type=access_type,
        )

    @classmethod
    def files(cls: Type["RocksDictStorage"], path: Path) -> Iterable[Path]:
        return (f for f in Path(path).glob("**/*") if f.is_file())

    def batch_read(
        self, keys: Iterable[HookComboKeyType]
    ) -> Sequence[HookComboValueType]:
        return [
            self.sr.deserialize(buffer)
            for buffer in self.db[[self.sr.key(k) for k in keys]]
        ]

    def batch_write(
        self,
        keys: Iterable[HookComboKeyType],
        values: Iterable[HookComboValueType],
    ) -> None:
        wb = WriteBatch(raw_mode=True)
        for key, array in zip(keys, values):
            wb.put(self.sr.key(key), self.sr.serialize(array))
        self.db.write(wb, write_opt=WriteOptions(sync=True))

    def batch_delete(self, keys: Iterable[HookComboKeyType]) -> None:
        wb = WriteBatch(raw_mode=True)
        [wb.delete(self.sr.key(key)) for key in keys]
        self.db.write(wb, write_opt=WriteOptions(sync=True))
