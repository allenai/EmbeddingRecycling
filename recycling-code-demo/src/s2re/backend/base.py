from abc import ABC, abstractclassmethod, abstractmethod
from functools import partial
from pathlib import Path
from typing import (
    Callable,
    Iterable,
    Optional,
    Protocol,
    Sequence,
    Type,
    Union,
    get_type_hints,
)

from ..types import BackendValueType, HookComboKeyType, HookComboValueType
from .serialization import PickleSerialization


class DBProtocol(Protocol):
    def __init__(self, path: Union[str, Path, None], *args, **kwargs):
        ...

    def close(self) -> None:
        ...


class BaseKVStorage(ABC):
    """An abstract implementation of a key-value storage,
    Subclasses must implement the following methods:
        - batch_write
        - batch_read
    """

    sr: PickleSerialization
    db: DBProtocol

    def __init__(
        self, path: Optional[Union[str, Path]] = None, read_only: bool = False
    ):
        """Initializes the storage. If path is None, data is stored in memory.
        If read only is True, the database is opened in read-only mode."""
        self.sr = PickleSerialization()
        self.path = Path(path) if path is not None else None
        self.read_only = read_only
        self.files = partial(self.files, path=self.path)  # type: ignore

    def write(self, key: HookComboKeyType, array: BackendValueType) -> None:
        """Write a single element to the storage"""
        return self.batch_write([key], [array])

    def read(self, key: HookComboKeyType) -> HookComboValueType:
        """Reads a single element from the storage"""
        return [
            e
            for e in self.batch_read(
                [
                    key,
                ]
            )
        ][0]

    def delete(self, key: HookComboKeyType) -> None:
        """Delete a single element from the storage"""
        return self.batch_delete((key,))

    def __exit__(self):
        """Close the database"""
        self.db.close()

    @abstractclassmethod
    def files(cls: "BaseKVStorage", path: Path) -> Iterable[Path]:
        """Returns a sequence of files making up this storage
        at location `path`"""
        raise NotImplementedError()

    @abstractmethod
    def batch_write(
        self,
        keys: Iterable[HookComboKeyType],
        values: Iterable[HookComboValueType],
    ) -> None:
        """Write a batch of elements to the storage."""
        raise NotImplementedError()

    @abstractmethod
    def batch_read(
        self, keys: Iterable[HookComboKeyType]
    ) -> Sequence[HookComboValueType]:
        """Reads a batch of elements from the storage; batch should be
        provided as a list of keys; each key can either be bytes or a string
        (strings will be encoded to bytes)."""
        raise NotImplementedError()

    @abstractmethod
    def batch_delete(self, keys: Iterable[HookComboKeyType]) -> None:
        raise NotImplementedError()


class BackendRegistry:
    __registry__ = {}

    def __new__(cls: Type["BackendRegistry"]) -> Type["BackendRegistry"]:
        return cls

    @classmethod
    def reg(cls: Type["BackendRegistry"], name: str) -> Callable:
        """Registers a subclass of BaseKVStorage under a given name"""
        return partial(cls.set, name=name)

    @classmethod
    def set(
        cls: Type["BackendRegistry"],
        backend_cls: Type[BaseKVStorage],
        name: str,
    ) -> Type[BaseKVStorage]:

        hints = get_type_hints(backend_cls)
        if hints.get("db", DBProtocol) == DBProtocol:
            raise TypeError(
                f"{backend_cls.__name__} must have a `db` "
                f"attribute with type hints"
            )
        if hints["db"] is None:
            return backend_cls

        if name in cls.__registry__:
            raise TypeError(f"Backend `{name}` already exists.")

        if not issubclass(backend_cls, BaseKVStorage):
            raise TypeError(f"`{name}` is not a subclass of BaseKVStorage.")

        cls.__registry__[name] = backend_cls
        return backend_cls

    @classmethod
    def all(cls: Type["BackendRegistry"]) -> Iterable[str]:
        return cls.__registry__.keys()

    @classmethod
    def get(cls: Type["BackendRegistry"], name: str) -> Type[BaseKVStorage]:
        """Returns the backend class corresponding to the given name"""
        try:
            return cls.__registry__[name]
        except KeyError:
            backends = ", ".join(f"`{b}`" for b in cls.all())
            raise TypeError(
                f"No backend with name `{name}`; "
                f"available backends: {backends}"
            )
