import multiprocessing
import threading
from collections import abc
from pathlib import Path
from queue import Queue
from time import sleep
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import numpy as np
import torch

from ..backend import BackendRegistry
from ..backend.base import BaseKVStorage
from ..types import HookComboKeyType, HookComboValueType


class StorageWrapper:
    """A wrapper for a storage backend.

    Takes care of moving tensors to cpu and removing gradients, including
    in the case of dictionaries and sequences of tensors.

    It should not be instantiated directly; instead, it is used by a
    CachingSession object.
    """

    # storage attribute will contain backend object
    storage: BaseKVStorage

    def __init__(
        self,
        backend: str,
        path: Union[str, Path],
        device: Union[str, torch.device],
        backend_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.device = device
        self.backend = backend
        self.backend_kwargs = backend_kwargs or {}
        self.path = path
        self.storage = self._get_storage()

    def _get_storage(self) -> BaseKVStorage:
        return BackendRegistry.get(self.backend)(
            path=self.path, **self.backend_kwargs
        )

    @classmethod
    def _cast_key(
        cls,
        key_to_cast: Union[
            torch.Tensor, Sequence[torch.Tensor], Sequence[bytes]
        ],
    ) -> List[bytes]:
        """Turns a tensor containing a key into a list of bytes."""
        is_seq = isinstance(key_to_cast, abc.Sequence)
        is_seq_bytes = is_seq and all(
            isinstance(elem, bytes) for elem in key_to_cast
        )

        if is_seq_bytes:
            # already a list of bytes
            return key_to_cast  # type: ignore

        seq_key = key_to_cast if is_seq else [key_to_cast]
        return [
            cls._cast_value(elem, device="cpu").numpy().tobytes()
            for elem in seq_key
        ]

    @staticmethod
    def _move_and_grad(
        tensor: torch.Tensor,
        grad: bool,
        device: Union[str, torch.device],
    ) -> torch.Tensor:
        """Moves a tensor to the device and attaches/detaches gradients.
        Sequence of gradients/device moving is slightly different depending
        on whether the gradients are getting attached or detached."""
        if not grad:
            tensor = tensor.detach()
        tensor = tensor.to(device)
        if grad:
            # NOTE: this is an in-place operation, no cloning
            tensor.requires_grad_(True)
        return tensor

    @classmethod
    def _cast_value(
        cls,
        value: HookComboValueType,
        device: Union[str, torch.device],
        requires_grad: bool = False,
    ) -> HookComboValueType:
        """Move tensor(s) to cpu, and remove gradients"""

        if isinstance(value, abc.Mapping):
            casted_value = {
                key: cls._cast_value(
                    element, requires_grad=requires_grad, device=device
                )
                for key, element in value.items()
            }
        elif isinstance(value, abc.Sequence):
            casted_value = [
                cls._cast_value(
                    element, requires_grad=requires_grad, device=device
                )
                for element in value
            ]
        elif isinstance(value, torch.Tensor):
            casted_value = cls._move_and_grad(
                value, grad=requires_grad, device=device
            )
        elif isinstance(value, np.ndarray):
            casted_value = torch.from_numpy(value)
        else:
            raise ValueError(f"Cannot cast value of type {type(value)}")

        return casted_value

    def store(
        self,
        key: Union[torch.Tensor, Sequence[torch.Tensor]],
        value: Union[torch.Tensor, Sequence[torch.Tensor]],
    ) -> None:
        """Stores a value or list of values in the storage backend."""

        seq_key = self._cast_key(key)
        seq_val = [value] if isinstance(key, torch.Tensor) else value

        casted_seq_val = self._cast_value(seq_val, device="cpu")
        self.storage.batch_write(keys=seq_key, values=casted_seq_val)

    def fetch(
        self: "StorageWrapper",
        key: Union[torch.Tensor, Sequence[torch.Tensor], Sequence[bytes]],
        training: bool = False,
    ) -> HookComboValueType:
        """Fetches values for one or more keys from the storage backend."""

        seq_val = [
            self._cast_value(v, device=self.device, requires_grad=training)
            for v in self.storage.batch_read(keys=self._cast_key(key))
        ]

        if isinstance(key, torch.Tensor):
            return seq_val[0]
        else:
            return seq_val

    def delete(
        self: "StorageWrapper",
        key: Union[torch.Tensor, Sequence[torch.Tensor], Sequence[bytes]],
    ) -> None:
        """Deletes one or a list of keys from the storage backend."""
        self.storage.batch_delete(keys=self._cast_key(seq_key))


class StopFlag:
    ...


class FetchAheadStorageWrapper(StorageWrapper):
    """A wrapper for a storage backend that prefetches data.

    This is a wrapper for StorageWrapper that prefetches data from
    the backend. It is used by CachingSession.

    It should not be instantiated directly; instead, it is used by a
    CachingSession object.
    """

    def __init__(
        self,
        backend: str,
        device: Union[str, torch.device],
        path: Union[str, Path],
        fetch_ahead: int,
        fetch_key_fn: Callable,
        backend_kwargs: Optional[Dict[str, Any]] = None,
        timeout: float = 0.1,
        retry_count: int = 10,
        new_queue_factory: Optional[Callable] = None,
        new_store_factory: Optional[Callable] = None,
    ):

        super().__init__(
            backend=backend,
            path=path,
            device=device,
            backend_kwargs=backend_kwargs,
        )

        if new_queue_factory is None:
            new_queue_factory = Queue
        if new_store_factory is None:
            new_store_factory = dict

        self._fetch_ahead = fetch_ahead

        # this is the queue that will hold the keys to be prefetched
        self._to_fetch_queue: Queue = new_queue_factory(maxsize=fetch_ahead)

        # this is the queue that will hold items that have been requested
        # in prefetching. We hold elements that should be yielded here.
        self._to_yield_queue: Queue = new_queue_factory()

        # this is the dictionary that will hold the prefetched data
        self._fetched_store: dict = new_store_factory()

        # time to sleep between queue checks
        self._fetch_timeout = timeout
        self._fetch_retry_count = retry_count
        self._fetch_key_fn = fetch_key_fn

        self._start_fetcher_thread()

    def _get_storage(self):
        # we return none here because we want to get the storage reader
        # in the fetching thread.
        ...

    def _start_fetcher_thread(
        self, new_thread_factory: Optional[Callable] = None
    ):
        new_thread_factory = new_thread_factory or threading.Thread

        # set up the thread that will prefetch data
        new_thread_factory(
            target=self._run_fetcher_thread,
            kwargs=dict(
                to_fetch_queue=self._to_fetch_queue,
                to_yield_queue=self._to_yield_queue,
                fetch_key_fn=self._fetch_key_fn,
                fetched_store=self._fetched_store,
                timeout=self._fetch_timeout,
                max_store_size=self._fetch_ahead,
                max_tries=self._fetch_retry_count,
                backend=self.backend,
                opts=self.backend_kwargs,
                path=self.path,
            ),
            # in case something goes wrong with data loader,
            # `daemon=True` will prevent process hanging
            daemon=True,
        ).start()

    @classmethod
    def _run_fetcher_thread(
        cls: Type["FetchAheadStorageWrapper"],
        to_fetch_queue: Queue[Any],
        to_yield_queue: Queue[Any],
        fetch_key_fn: Callable[[Any], HookComboKeyType],
        fetched_store: Dict[Tuple[bytes], Any],
        timeout: float,
        max_tries: int,
        backend: str,
        opts: Optional[Dict[str, Any]],
        path: Union[str, Path],
        max_store_size: int = 0,
    ) -> None:

        storage_wrapper = StorageWrapper(
            backend=backend, path=path, backend_kwargs=opts, device="cpu"
        )

        while True:
            elem = to_fetch_queue.get(block=True, timeout=timeout * max_tries)

            if isinstance(elem, StopFlag):
                to_yield_queue.put(StopFlag(), block=True)
                break

            key = fetch_key_fn(elem)
            cast_key = cls._cast_key(key)
            value = storage_wrapper.fetch(cast_key)

            max_tries_cnt = max_tries
            while max_store_size > 0 and len(fetched_store) >= max_store_size:
                sleep(timeout)
                max_tries_cnt -= 1

                if max_tries_cnt <= 0:
                    raise ValueError(f"Failed to prefetch {key}")

            fetched_store[tuple(cast_key)] = value
            to_yield_queue.put(elem, block=True, timeout=timeout * max_tries)

        del storage_wrapper

    def _start_key_reader_thread(
        self,
        iterable: Iterable[Any],
        new_thread_factory: Optional[Callable] = None,
    ):
        new_thread_factory = new_thread_factory or threading.Thread

        # set up the thread that will read keys from the fetched store
        new_thread_factory(
            target=self._run_key_reader_thread,
            kwargs=dict(
                iterable=iterable,
                to_fetch_queue=self._to_fetch_queue,
                max_tries=self._fetch_retry_count,
                timeout=self._fetch_timeout,
            ),
            # in case something goes wrong with data loader,
            # `daemon=True` will prevent process hanging
            daemon=True,
        ).start()

    @classmethod
    def _run_key_reader_thread(
        cls: Type["FetchAheadStorageWrapper"],
        iterable: Iterable[HookComboKeyType],
        max_tries: int,
        to_fetch_queue: Queue[Union[HookComboKeyType, StopFlag]],
        timeout: float,
    ) -> None:

        for elem in iterable:
            # wait till the queue has a spot to set the element
            # for prefetching.
            to_fetch_queue.put(elem, block=True, timeout=timeout * max_tries)

        to_fetch_queue.put(StopFlag())

    def prefetch(self, iterable: Iterable[Any]) -> Iterable[Any]:
        self._start_key_reader_thread(iterable=iterable)

        while True:
            elem = self._to_yield_queue.get(
                block=True,
                timeout=self._fetch_timeout * self._fetch_retry_count,
            )
            if isinstance(elem, StopFlag):
                break
            else:
                yield elem

    def fetch(
        self: "FetchAheadStorageWrapper",
        key: Union[torch.Tensor, Sequence[torch.Tensor]],
    ) -> HookComboValueType:
        """Fetches values for one or more keys from the storage backend."""

        # casting Tensors to bytes, getting the values.
        seq_key = tuple(self._cast_key(key))

        retry_count = self._fetch_retry_count

        while seq_key not in self._fetched_store:
            if retry_count > 0:
                sleep(self._fetch_timeout)

                # decrease the retry count
                retry_count -= 1
            else:
                # we have tried enough times, so we give up
                raise KeyError(f"Timeout: key {seq_key} not found in storage")

        seq_val = self._fetched_store.pop(seq_key)

        seq_val = [self._cast_value(v, device=self.device) for v in seq_val]

        if isinstance(key, torch.Tensor):
            return seq_val[0]
        else:
            return seq_val

    def store(self, *_, **__) -> None:
        raise NotImplementedError("Cannot write to storage in prefetch mode.")

    def delete(self, *_, **__) -> None:
        raise NotImplementedError(
            "Cannot delete from storage in prefetch mode."
        )


class MultiprocessingFetchAheadStorageWrapper(FetchAheadStorageWrapper):
    def __init__(self, *args, **kwargs):
        self.manager = multiprocessing.Manager()
        super().__init__(
            *args,
            **kwargs,
            new_queue_factory=self.manager.Queue,
            new_store_factory=self.manager.dict,
        )

    def _start_fetcher_thread(self):
        return super()._start_fetcher_thread(
            new_thread_factory=multiprocessing.Process
        )

    def _start_key_reader_thread(self, iterable: Iterable[Any]):
        return super()._start_key_reader_thread(
            iterable=iterable, new_thread_factory=multiprocessing.Process
        )
