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
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    overload,
)

import numpy as np
import torch

from ..backend import BackendRegistry
from ..backend.base import BaseKVStorage
from ..types import HookComboKeyType, HookComboValueType

BaseKeyType = Union[torch.Tensor, Sequence[torch.Tensor], Sequence[bytes]]


class MoveAndGradMixIn:
    @staticmethod
    def _move_and_grad(
        tensor: torch.Tensor,
        grad: bool,
        device: Union[str, torch.device],
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Moves a tensor to the device and attaches/detaches gradients.
        Sequence of gradients/device moving is slightly different depending
        on whether the gradients are getting attached or detached."""
        if not grad:
            tensor = tensor.detach()

        if dtype:
            tensor = tensor.type(dtype)
        tensor = tensor.to(device)

        if grad:
            # NOTE: this is an in-place operation, no cloning
            tensor.requires_grad_(True)
        return tensor


class KeyCastingMixIn(MoveAndGradMixIn):
    @overload
    @classmethod
    def _cast_key(
        cls,
        key_to_cast: torch.Tensor,
        cast_type_map: Optional[Mapping[torch.dtype, torch.dtype]] = None,
    ) -> Sequence[bytes]:
        ...

    @overload
    @classmethod
    def _cast_key(
        cls,
        key_to_cast: Sequence[torch.Tensor],
        cast_type_map: Optional[Mapping[torch.dtype, torch.dtype]] = None,
    ) -> Sequence[bytes]:
        ...

    @overload
    @classmethod
    def _cast_key(
        cls,
        key_to_cast: Sequence[bytes],
        cast_type_map: Optional[Mapping[torch.dtype, torch.dtype]] = None,
    ) -> Sequence[bytes]:
        ...

    @classmethod
    def _cast_key(
        cls,
        key_to_cast: BaseKeyType,
        cast_type_map: Optional[Mapping[torch.dtype, torch.dtype]] = None,
    ) -> Sequence[bytes]:
        """Turns a tensor containing a key into a list of bytes."""
        if isinstance(key_to_cast, torch.Tensor):
            key_to_cast = [key_to_cast]

        casted_key = [
            cls._move_and_grad(
                tensor=key,
                grad=False,
                device="cpu",
                dtype=(
                    cast_type_map.get(key.dtype, key.dtype)
                    if cast_type_map
                    else None
                ),
            )
            .numpy()
            .tobytes()
            if not isinstance(key, bytes)
            else key
            for key in key_to_cast
        ]
        return casted_key


BaseValueType = Union[torch.Tensor, np.ndarray]
AllValueContainersToCast = Union[
    BaseValueType,
    Sequence[BaseValueType],
    Mapping[str, BaseValueType],
    Sequence[Sequence[BaseValueType]],
    Sequence[Mapping[str, BaseValueType]],
]
AllValueContainersCasted = Union[
    torch.Tensor,
    Mapping[str, Union[torch.Tensor, Sequence[torch.Tensor]]],
    Sequence[
        Union[
            torch.Tensor, Sequence[torch.Tensor], Mapping[str, torch.Tensor]
        ]
    ],
]


class ValueCastingMixIn(MoveAndGradMixIn):
    @overload
    @classmethod
    def _cast_value(
        cls,
        value: BaseValueType,
        device: Union[str, torch.device],
        requires_grad: bool = False,
        cast_type_map: Optional[Mapping[torch.dtype, torch.dtype]] = None,
    ) -> torch.Tensor:
        ...

    @overload
    @classmethod
    def _cast_value(
        cls,
        value: Mapping[str, BaseValueType],
        device: Union[str, torch.device],
        requires_grad: bool = False,
        cast_type_map: Optional[Mapping[torch.dtype, torch.dtype]] = None,
    ) -> Mapping[str, torch.Tensor]:
        ...

    @overload
    @classmethod
    def _cast_value(
        cls,
        value: Sequence[BaseValueType],
        device: Union[str, torch.device],
        requires_grad: bool = False,
        cast_type_map: Optional[Mapping[torch.dtype, torch.dtype]] = None,
    ) -> Sequence[torch.Tensor]:
        ...

    @overload
    @classmethod
    def _cast_value(
        cls,
        value: Sequence[Sequence[BaseValueType]],
        device: Union[str, torch.device],
        requires_grad: bool = False,
        cast_type_map: Optional[Mapping[torch.dtype, torch.dtype]] = None,
    ) -> Sequence[Sequence[torch.Tensor]]:
        ...

    @overload
    @classmethod
    def _cast_value(
        cls,
        value: Sequence[Mapping[str, BaseValueType]],
        device: Union[str, torch.device],
        requires_grad: bool = False,
        cast_type_map: Optional[Mapping[torch.dtype, torch.dtype]] = None,
    ) -> Sequence[Mapping[str, torch.Tensor]]:
        ...

    @classmethod
    def _cast_value(
        cls,
        value: AllValueContainersToCast,
        device: Union[str, torch.device],
        requires_grad: bool = False,
        cast_type_map: Optional[Mapping[torch.dtype, torch.dtype]] = None,
    ) -> AllValueContainersCasted:
        """Move tensor(s) to cpu, and remove gradients"""

        casted_value: AllValueContainersCasted

        if isinstance(value, abc.Mapping):
            casted_value = {
                key: cls._cast_value(
                    value=element,
                    requires_grad=requires_grad,
                    device=device,
                    cast_type_map=cast_type_map,
                )
                for key, element in value.items()
            }
        elif isinstance(value, abc.Sequence):
            casted_valued = [
                cls._cast_value(
                    element,
                    requires_grad=requires_grad,
                    device=device,
                    cast_type_map=cast_type_map,
                )
                for element in value
            ]
            casted_value = casted_valued
        elif isinstance(value, torch.Tensor):
            casted_type = (
                cast_type_map.get(value.dtype, None)
                if cast_type_map is not None
                else None
            )
            casted_value = cls._move_and_grad(
                value, grad=requires_grad, device=device, dtype=casted_type
            )
        elif isinstance(value, np.ndarray):
            casted_value = torch.from_numpy(value)
            if (
                cast_type_map is not None
                and (dtype := cast_type_map.get(casted_value.dtype, None))
                is not None
            ):
                casted_value = casted_value.type(dtype)

        else:
            raise ValueError(f"Cannot cast value of type {type(value)}")

        return casted_value


ValueStorageWrapperType = Union[
    torch.Tensor, Sequence[torch.Tensor], Mapping[str, torch.Tensor]
]


class StorageWrapper(KeyCastingMixIn, ValueCastingMixIn):
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
        cast_types_map: Optional[Dict[torch.dtype, torch.dtype]] = None,
    ):
        self.device = device
        self.backend = backend
        self.backend_kwargs = backend_kwargs or {}
        self.path = path
        self.storage = self._get_storage()
        self.cast_types_map = cast_types_map

    def _get_storage(self) -> BaseKVStorage:
        return BackendRegistry.get(self.backend)(
            path=self.path, **self.backend_kwargs
        )

    @overload
    def store(
        self,
        key: torch.Tensor,
        value: ValueStorageWrapperType,
    ) -> None:
        ...

    @overload
    def store(
        self,
        key: Sequence[torch.Tensor],
        value: Sequence[ValueStorageWrapperType],
    ) -> None:
        ...

    def store(
        self,
        key: Union[torch.Tensor, Sequence[torch.Tensor]],
        value: Union[
            ValueStorageWrapperType, Sequence[ValueStorageWrapperType]
        ],
    ) -> None:
        """Stores a value or list of values in the storage backend."""

        seq_key = self._cast_key(key, cast_type_map=self.cast_types_map)

        # always wrapping single tensors otherwise they don't match with
        # casted_key
        if isinstance(key, torch.Tensor):
            seq_val: Sequence[ValueStorageWrapperType] = [value]  # type: ignore
        else:
            seq_val: Sequence[ValueStorageWrapperType] = value  # type: ignore

        casted_seq_val = self._cast_value(
            value=seq_val, device="cpu", cast_type_map=self.cast_types_map
        )

        self.storage.batch_write(keys=seq_key, values=casted_seq_val)

    def fetch(
        self: "StorageWrapper",
        key: Union[torch.Tensor, Sequence[torch.Tensor], Sequence[bytes]],
        training: bool = False,
    ) -> AllValueContainersCasted:
        """Fetches values for one or more keys from the storage backend."""

        seq_val = [
            self._cast_value(
                value=value,
                device=self.device,
                requires_grad=training,
                cast_type_map=self.cast_types_map,
            )
            for value in self.storage.batch_read(
                keys=self._cast_key(key, cast_type_map=self.cast_types_map)
            )
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
        self.storage.batch_delete(
            keys=self._cast_key(key, cast_type_map=self.cast_types_map)
        )


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
        cast_types_map: Optional[Dict[torch.dtype, torch.dtype]] = None,
        new_queue_factory: Optional[Callable] = None,
        new_store_factory: Optional[Callable] = None,
    ):

        super().__init__(
            backend=backend,
            path=path,
            device=device,
            backend_kwargs=backend_kwargs,
            cast_types_map=cast_types_map,
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
                cast_type_map=self.cast_types_map,
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
        fetch_key_fn: Callable[[Any], BaseKeyType],
        fetched_store: Dict[Tuple[bytes], Any],
        timeout: float,
        max_tries: int,
        backend: str,
        opts: Optional[Dict[str, Any]],
        path: Union[str, Path],
        max_store_size: int = 0,
        cast_type_map: Optional[Mapping[torch.dtype, torch.dtype]] = None,
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
            cast_key = cls._cast_key(key, cast_type_map=cast_type_map)
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
        seq_key = tuple(
            self._cast_key(key, cast_type_map=self.cast_types_map)
        )

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

        seq_val = [
            self._cast_value(
                v, device=self.device, cast_type_map=self.cast_types_map
            )
            for v in seq_val
        ]

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
