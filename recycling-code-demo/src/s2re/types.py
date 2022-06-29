from typing import List, Union, Dict, TypeVar

from numpy import ndarray
from torch import Tensor


HookSingleKeyType = Union[Tensor, bytes]
HookComboKeyType = Union[HookSingleKeyType, List[HookSingleKeyType]]

BackendValueType = Union[Tensor, ndarray]
HookSingleValueType = Union[BackendValueType,
                            List[BackendValueType],
                            Dict[str, BackendValueType]]
HookComboValueType = Union[
    BackendValueType,
    List[BackendValueType],
    Dict[str, BackendValueType],
    List[List[BackendValueType]],
    List[Dict[str, BackendValueType]]
]
