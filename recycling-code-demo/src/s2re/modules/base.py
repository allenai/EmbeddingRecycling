from abc import ABC, abstractmethod
from typing import Tuple, Union

import torch

from ..context.session import CachingSession


class BaseModuleWithCaching(ABC, torch.nn.Module):
    @property
    def cache(self) -> Union[CachingSession, None]:
        """Get the caching session object."""
        return getattr(self, "__caching_session__", None)

    def set_session(self, caching_session: CachingSession):
        """Assign a session object to the module."""
        return setattr(self, "__caching_session__", caching_session)

    def del_session(self):
        """Remove the session object from the module."""
        return delattr(self, "__caching_session__")

    def get_cache_hit_value(self):
        """Value to return when there is a cache hit.
        Subclasses might customize it for compatibility reasons."""
        return None


class CacheKeyLookup(BaseModuleWithCaching):
    def _find_cache_key(self, *args, **kwargs) -> torch.LongTensor:
        name_or_pos = self.get_cache_arg_name_or_pos()
        if isinstance(name_or_pos, tuple):
            name, pos = name_or_pos
        elif isinstance(name_or_pos, str):
            name, pos = name_or_pos, None
        elif isinstance(name_or_pos, int):
            pos, name = name_or_pos, None
        else:
            pos = name = None

        if name is not None:
            if name not in kwargs and pos is None:
                raise ValueError(f"Keyword argument {name} not found.")
            return kwargs[name]

        elif pos is not None:
            if pos >= len(args):
                raise ValueError(f"Positional argument {pos} not found.")
            return args[pos]

        else:
            raise ValueError("Either name or position must be specified.")

    @abstractmethod
    def get_cache_arg_name_or_pos(self) -> Union[str, int, Tuple[str, int]]:
        """Name and/or position of the argument to use for caching.

        Examples:

        - `get_cache_arg_name_or_pos(self): return 'input_ids'`:  Look for a
            keyword argument called `input_ids`, and use that for caching.
        - `get_cache_arg_name_or_pos(self): return 0`:  Use the  first
            positional argument for caching.
        - `get_cache_arg_name_or_pos(self): return ('input_ids', 0)`:  Use
            the keyword argument `input_ids` for caching; `input_ids` is not
            found, use the first positional argument.
        """
        raise NotImplementedError(
            "Subclasses must implement this method "
            "to support setting the cache key."
        )

    def forward(self, *args, **kwargs):
        if self.cache is None:
            return super().forward(*args, **kwargs)
        elif self.cache.recording:
            value = super().forward(*args, **kwargs)
        else:
            value = self.get_cache_hit_value()

        self.cache.key(self._find_cache_key(*args, **kwargs))
        return value


class NoOpWhenCached(BaseModuleWithCaching):
    def forward(self, *args, **kwargs):
        if self.cache is None or self.cache.recording:
            return super().forward(*args, **kwargs)
        else:
            return self.get_cache_hit_value()


class CachedLayer(BaseModuleWithCaching):
    def forward(self, *args, **kwargs):
        if self.cache is None:
            return super().forward(*args, **kwargs)
        elif self.cache.recording:
            out = super().forward(*args, **kwargs)
            self.cache.store(out)
        else:
            out = self.cache.fetch()
        return out
