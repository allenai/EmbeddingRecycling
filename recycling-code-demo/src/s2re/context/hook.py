from contextlib import contextmanager
from inspect import getfullargspec, unwrap
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    Optional,
    Sequence,
    Type,
    Union,
)

import torch

from ..backend import BackendRegistry
from ..modules.base import BaseModuleWithCaching
from .session import CachingSession


class CachingHook:
    """Hook to start and stop caching session."""

    def __init__(self, **session_kwargs: Any):
        self.session_kwargs = session_kwargs

    @staticmethod
    def available_backends() -> Sequence[str]:
        """Return a list of available backends."""
        return list(BackendRegistry.all())

    @staticmethod
    def find_all_caching_modules(
        module: torch.nn.Module,
    ) -> Sequence[BaseModuleWithCaching]:
        out = [
            m
            for _, m in module.named_modules()
            if isinstance(m, BaseModuleWithCaching)
        ]
        if isinstance(module, BaseModuleWithCaching):
            out.insert(0, module)
        return out

    @staticmethod
    def infer_device(module: torch.nn.Module) -> torch.device:
        device = {p.device for p in module.parameters()}
        if len(device) == 1:
            return device.pop()
        raise ValueError(
            f"Module {module} has parameters on multiple"
            f"({len(device):,}) devices; provide device "
            "manually."
        )

    @classmethod
    @contextmanager
    def Record(
        cls: Type["CachingHook"],
        module: torch.nn.Module,
        backend: str,
        path: Union[str, Path],
        backend_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ) -> Iterator[CachingSession]:
        caching_modules = cls.find_all_caching_modules(module)

        session = CachingSession(
            recording=True,
            training=False,
            device=device or cls.infer_device(module),
            backend=backend,
            path=path,
            backend_kwargs=backend_kwargs,
        )
        try:
            [m.set_session(session) for m in caching_modules]
            yield session
        finally:
            [m.del_session() for m in caching_modules]

    @classmethod
    @contextmanager
    def Use(
        cls: Type["CachingHook"],
        module: torch.nn.Module,
        backend: str,
        path: Union[str, Path],
        backend_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        fetch_ahead: int = -1,
        fetch_spawn: str = "thread",
        fetch_key_fn: Optional[Callable] = None,
        fetch_timeout: float = 0.1,
        fetch_retry_count: int = 10,
    ) -> Iterator[CachingSession]:

        caching_modules = cls.find_all_caching_modules(module)
        session = CachingSession(
            recording=False,
            training=False,
            device=device or cls.infer_device(module),
            backend=backend,
            fetch_spawn=fetch_spawn,
            path=path,
            backend_kwargs=backend_kwargs,
            fetch_ahead=fetch_ahead,
            fetch_key_fn=fetch_key_fn,
            fetch_timeout=fetch_timeout,
            fetch_retry_count=fetch_retry_count,
        )
        try:
            [m.set_session(session) for m in caching_modules]
            yield session
        finally:
            [m.del_session() for m in caching_modules]

    @classmethod
    @contextmanager
    def Train(
        cls: Type["CachingHook"],
        module: torch.nn.Module,
        backend: str,
        path: Union[str, Path],
        backend_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        fetch_ahead: int = -1,
        fetch_spawn: str = "thread",
        fetch_key_fn: Optional[Callable] = None,
        fetch_timeout: float = 0.1,
        fetch_retry_count: int = 10,
    ) -> Iterator[CachingSession]:

        caching_modules = cls.find_all_caching_modules(module)
        session = CachingSession(
            recording=False,
            training=True,
            device=device or cls.infer_device(module),
            backend=backend,
            fetch_spawn=fetch_spawn,
            path=path,
            backend_kwargs=backend_kwargs,
            fetch_ahead=fetch_ahead,
            fetch_key_fn=fetch_key_fn,
            fetch_timeout=fetch_timeout,
            fetch_retry_count=fetch_retry_count,
        )
        try:
            [m.set_session(session) for m in caching_modules]
            yield session
        finally:
            [m.del_session() for m in caching_modules]

    def _check_spec(
        self,
        method_name: str,
        session_args: Sequence[Any],
        session_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        fn_spec = getfullargspec(unwrap(getattr(self, method_name)))
        args_except_cls = fn_spec.args[1:]

        # we want to get the list of required arguments, which involves
        # getting the count of non-default arguments and hen slicing
        # the list of method arguments.
        cnt_required = len(args_except_cls) - (
            len(fn_spec.defaults) if fn_spec.defaults else 0
        )
        required = set(args_except_cls[:cnt_required])

        # if any argument is provided, we turn into a dictionary that we
        # can merge with the remaining kwargs.
        session_args_dict = dict(zip(args_except_cls, session_args))

        # merge all three sources of kwargs: the keyword arguments provided at
        # object creation, the keyword arguments provided at method invocation,
        # and the positional arguments we just turned into a dictionary.
        merged_session_kwargs = {
            **self.session_kwargs,
            **session_kwargs,
            **session_args_dict,
        }

        # we filter the arguments so that only the ones this method accepts
        # are kept.
        filtered_session_kwargs = {
            k: v
            for k, v in merged_session_kwargs.items()
            if k in args_except_cls
        }

        # finally, we check which arguments are missing.
        if not all(k in filtered_session_kwargs for k in required):
            raise ValueError(f"{method_name} requires {required}")

        # if none is missing, we return the filtered kwargs.
        return filtered_session_kwargs

    @contextmanager
    def record(
        self, *session_args: Any, **session_kwargs: Any
    ) -> Iterator[CachingSession]:
        filtered_session_kwargs = self._check_spec(
            "Record", session_args, session_kwargs
        )
        try:
            with self.Record(**filtered_session_kwargs) as session:
                yield session
        finally:
            ...

    @contextmanager
    def use(
        self, *session_args: Any, **session_kwargs: Any
    ) -> Iterator[CachingSession]:
        filtered_session_kwargs = self._check_spec(
            "Use", session_args, session_kwargs
        )
        try:
            with self.Use(**filtered_session_kwargs) as session:
                yield session
        finally:
            ...

    @contextmanager
    def train(
        self, *session_args: Any, **session_kwargs: Any
    ) -> Iterator[CachingSession]:
        filtered_session_kwargs = self._check_spec(
            "Train", session_args, session_kwargs
        )
        try:
            with self.Train(**filtered_session_kwargs) as session:
                yield session
        finally:
            ...
