

import pickle
from collections import abc
from enum import IntFlag
from typing import NamedTuple, Union, Mapping, Sequence
from regex import F

import torch

from typing import Type

from ..types import BackendValueType, HookSingleValueType


class CONTAINER_TYPE(IntFlag):
    """Enum for container types."""
    DICT = 1
    LIST = 2
    NUMPY = 3
    TORCH = 4


class SerializationContainer(NamedTuple):
    """Flags for serialization."""
    type: CONTAINER_TYPE
    data: Union[BackendValueType, 'SerializationContainer']


class PickleSerialization:
    @classmethod
    def _raw_load(
        cls: Type['PickleSerialization'],
        buffer: bytes
    ) -> SerializationContainer:
        return pickle.loads(buffer)

    @classmethod
    def _raw_dump(
        cls: Type['PickleSerialization'],
        container: SerializationContainer
    ) -> bytes:
        return pickle.dumps(container)

    @classmethod
    def _prepare_serialize_array(
        cls: Type['PickleSerialization'],
        array: BackendValueType
    ) -> SerializationContainer:
        if isinstance(array, torch.Tensor):
            array = array.detach().cpu().numpy()
            type_ = CONTAINER_TYPE.TORCH
        else:
            type_ = CONTAINER_TYPE.NUMPY

        return SerializationContainer(data=array, type=type_)

    @classmethod
    def _deserialize_array(
        cls: Type['PickleSerialization'],
        container: SerializationContainer
    ) -> BackendValueType:
        array = container.data
        if container.type == CONTAINER_TYPE.TORCH:
            array = torch.from_numpy(array)
        return array

    @classmethod
    def _prepare_serialize_dict(
        cls: Type['PickleSerialization'],
        di: Mapping[str, BackendValueType]
    ) -> SerializationContainer:
        di = {k: cls._prepare_serialize_array(v) for k, v in di.items()}
        return SerializationContainer(data=di, type=CONTAINER_TYPE.DICT)

    @classmethod
    def _deserialize_dict(
        cls: Type['PickleSerialization'],
        container: SerializationContainer
    ) -> Mapping[str, BackendValueType]:
        di = {k: cls._deserialize_array(v) for k, v in container.data.items()}
        return di

    @classmethod
    def _prepare_serialize_sequence(
        cls: Type['PickleSerialization'],
        seq: Sequence[BackendValueType]
    ) -> SerializationContainer:
        seq = [cls._prepare_serialize_array(v) for v in seq]
        return SerializationContainer(data=seq, type=CONTAINER_TYPE.LIST)

    @classmethod
    def _deserialize_sequence(
        cls: Type['PickleSerialization'],
        container: SerializationContainer
    ) -> Sequence[BackendValueType]:
        seq = [cls._deserialize_array(v) for v in container.data]
        return seq

    @classmethod
    def serialize(
        cls: Type['PickleSerialization'],
        value: HookSingleValueType
    ) -> bytes:
        if isinstance(value, abc.Mapping):
            container = cls._prepare_serialize_dict(value)
        elif isinstance(value, abc.Sequence):
            container = cls._prepare_serialize_sequence(value)
        else:
            container = cls._prepare_serialize_array(value)
        return cls._raw_dump(container)

    @classmethod
    def deserialize(
        cls: Type['PickleSerialization'],
        buffer: bytes
    ) -> HookSingleValueType:
        container = cls._raw_load(buffer)
        if container.type == CONTAINER_TYPE.DICT:
            value = cls._deserialize_dict(container)
        elif container.type == CONTAINER_TYPE.LIST:
            value = cls._deserialize_sequence(container)
        else:
            value = cls._deserialize_array(container)
        return value

    @classmethod
    def key(cls, key: bytes) -> bytes:
        if isinstance(key, str):
            key = key.encode('utf-8')
        return key
