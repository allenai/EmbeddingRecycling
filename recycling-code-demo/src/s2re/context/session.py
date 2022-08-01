from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Union

import torch

from ..types import HookComboValueType
from .wrapper import (
    FetchAheadStorageWrapper,
    MultiprocessingFetchAheadStorageWrapper,
    StorageWrapper,
)


class CachingSession:
    """An object that contains tools to work in a caching session:
    the function to store output after computation, or the function
    to retrieve the cached output.

    Calling the `recording` attribute tells which mode this caching
    session is running: if True, it is set to record partial
    computations; if False, partial computations are retrieved instead
    of re-running layers.
    """

    def __init__(
        self,
        recording: bool,
        backend: str,
        device: Union[str, torch.device],
        path: Union[str, Path],
        training: bool = False,
        backend_kwargs: Optional[Dict[str, Any]] = None,
        fetch_ahead: int = -1,
        fetch_key_fn: Optional[Callable] = None,
        fetch_spawn: str = "thread",
        fetch_timeout: float = 0.1,
        fetch_retry_count: int = 10,
        half_precision: bool = False,
    ):
        self._key = None

        self.recording = recording
        self.training = training

        if half_precision:
            cast_type_map = {
                torch.float32: torch.float16,
                torch.float64: torch.float16,
                torch.int64: torch.int16,
                torch.int32: torch.int16,
            }
        else:
            cast_type_map = None

        if self.recording and self.training:
            raise ValueError(
                "Can not record embeddings " "in cache while training"
            )

        if fetch_ahead > 0:
            assert (
                fetch_key_fn is not None
            ), "fetch_key_fn must be provided when prefetching"

            if fetch_spawn == "process":
                wrapper_factory = MultiprocessingFetchAheadStorageWrapper
            elif fetch_spawn == "thread":
                wrapper_factory = FetchAheadStorageWrapper
            else:
                raise ValueError(
                    f"Unknown fetch_spawn value: {fetch_spawn} "
                    '(expected "process" or "thread")'
                )

            self.storage = wrapper_factory(
                backend=backend,
                device=device,
                path=path,
                fetch_ahead=fetch_ahead,
                backend_kwargs=backend_kwargs,
                fetch_key_fn=fetch_key_fn,
                timeout=fetch_timeout,
                retry_count=fetch_retry_count,
                cast_types_map=cast_type_map,
            )
        else:
            self.storage = StorageWrapper(
                backend=backend,
                path=path,
                backend_kwargs=backend_kwargs,
                device=device,
                cast_types_map=cast_type_map,
            )

    def iterate(self, iterable: Iterable[Any]) -> Iterable[Any]:
        if isinstance(self.storage, FetchAheadStorageWrapper):
            yield from self.storage.prefetch(iterable)
        else:
            yield from iterable

    def store(self, value: torch.Tensor):
        if self._key is None:
            raise RuntimeError("Key not provided")

        if not self.recording:
            raise RuntimeError("Not in cache recording mode!")

        self.storage.store(key=self._key, value=value)
        self._key = None

    def fetch(self) -> HookComboValueType:
        if self._key is None:
            raise RuntimeError("Key not provided")

        if self.recording:
            raise RuntimeError("Not in cache fetching mode!")

        out = self.storage.fetch(self._key)

        self._key = None
        return out

    def key(self, key: torch.Tensor):
        if self._key is not None:
            raise RuntimeError("Key is already set")
        self._key = key
