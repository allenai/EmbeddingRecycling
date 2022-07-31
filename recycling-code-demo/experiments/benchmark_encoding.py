import io
import pickle
import pickletools
import sys
import time
import warnings
from abc import ABC, abstractstaticmethod
from pathlib import Path

import numpy as np
import pyarrow
import torch

try:
    from s2re.backend.serialization import PickleSerialization
except ImportError:
    src = Path(__file__).parent / ".." / "src"
    sys.path.append(str(src))
    from s2re.backend.serialization import PickleSerialization


class Benchmark(ABC):
    BENCHMARKS = []

    @abstractstaticmethod
    def save(t: torch.Tensor) -> bytes:
        ...

    @abstractstaticmethod
    def load(b: bytes) -> torch.Tensor:
        ...

    @classmethod
    def register(cls, benchmark_cls):
        if benchmark_cls not in cls.BENCHMARKS:
            cls.BENCHMARKS.append(benchmark_cls)
        else:
            raise ValueError(f"{benchmark_cls} is already registered")
        return cls

    @classmethod
    def benchmarks(cls):
        yield from cls.BENCHMARKS


@Benchmark.register
class LibImplementation(Benchmark):
    @staticmethod
    def save(t: torch.Tensor) -> bytes:
        return PickleSerialization.serialize(t)

    @staticmethod
    def load(b: bytes) -> torch.Tensor:
        return PickleSerialization.deserialize(b)


@Benchmark.register
class ViaPickleWithBuffers(Benchmark):
    @staticmethod
    def save(t: torch.Tensor) -> bytes:
        buffers = []
        data = pickle.dumps(
            t.detach().cpu().numpy(),
            protocol=5,
            buffer_callback=buffers.append,
        )
        return pickle.dumps((data, *(bytes(b.raw()) for b in buffers)))

    @staticmethod
    def load(bytes_data: bytes) -> torch.Tensor:
        data, *buffers = pickle.loads(bytes_data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return torch.from_numpy(
                pickle.loads(data, buffers=buffers)
            ).clone()


@Benchmark.register
class ViaNumpy(Benchmark):
    @staticmethod
    def save(t: torch.Tensor) -> bytes:
        array = t.cpu().detach().numpy()
        return pickle.dumps((array.tobytes(), array.shape, array.dtype))

    @staticmethod
    def load(b: bytes) -> torch.Tensor:
        data, shape, dtype = pickle.loads(b)
        array = np.ndarray(shape=shape, buffer=data, dtype=dtype)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return torch.from_numpy(array).clone()


@Benchmark.register
class ViaBuffer(Benchmark):
    @staticmethod
    def save(t: torch.Tensor) -> bytes:
        buffer = io.BytesIO()
        torch.save(
            t.cpu().detach(),
            buffer,
            pickle_protocol=5,
            _use_new_zipfile_serialization=False,
        )
        return buffer.getvalue()

    @staticmethod
    def load(b: bytes) -> torch.Tensor:
        buffer = io.BytesIO(b)
        return torch.load(buffer)


@Benchmark.register
class ViaBufferNumpy(Benchmark):
    @staticmethod
    def save(t: torch.Tensor) -> bytes:
        buffer = io.BytesIO()
        np.save(buffer, t.cpu().detach().numpy())
        return buffer.getvalue()

    @staticmethod
    def load(b: bytes) -> torch.Tensor:
        buffer = io.BytesIO(b)
        arr = np.load(buffer)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return torch.from_numpy(arr).clone()


@Benchmark.register
class ViaPyArrow(Benchmark):
    @staticmethod
    def save(t: torch.Tensor) -> bytes:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pyarrow.serialize(t.detach().cpu().numpy()).to_buffer()

    @staticmethod
    def load(b: bytes) -> torch.Tensor:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return torch.from_numpy(pyarrow.deserialize(b)).clone()


@Benchmark.register
class ViaPickleTorch(Benchmark):
    @staticmethod
    def save(t: torch.Tensor) -> bytes:
        return pickle.dumps(t.detach().cpu(), protocol=5)

    @staticmethod
    def load(b: bytes) -> torch.Tensor:
        return pickle.loads(b)


@Benchmark.register
class ViaPickleNumpy(Benchmark):
    @staticmethod
    def save(t: torch.Tensor) -> bytes:
        return pickle.dumps((1, t.detach().cpu().numpy()), protocol=5)

    @staticmethod
    def load(b: bytes) -> torch.Tensor:
        _, array = pickle.loads(b)
        return torch.from_numpy(array)


@Benchmark.register
class ViaPickleNumpyOptimize(Benchmark):
    @staticmethod
    def save(t: torch.Tensor) -> bytes:
        dump = pickle.dumps(t.detach().cpu().numpy(), protocol=5)
        return pickletools.optimize(dump)

    @staticmethod
    def load(b: bytes) -> torch.Tensor:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return torch.from_numpy(pickle.loads(b)).clone()


@Benchmark.register
class PyArrowTest2(Benchmark):
    @staticmethod
    def save(t: torch.Tensor) -> bytes:
        sink = pyarrow.BufferOutputStream()
        torch.save(t.detach().cpu(), sink)
        return sink.getvalue()

    @staticmethod
    def load(b: bytes) -> torch.Tensor:
        buffer = io.BytesIO(b)
        return torch.load(buffer)


@Benchmark.register
class ViaNumpyAndTorch(Benchmark):
    @staticmethod
    def save(t: torch.Tensor) -> bytes:
        array = t.cpu().detach().numpy()
        return pickle.dumps((array.tobytes(), t.shape, t.dtype))

    @staticmethod
    def load(b: bytes) -> torch.Tensor:
        data, shape, dtype = pickle.loads(b)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return torch.frombuffer(data, dtype=dtype).clone().view(shape)


@Benchmark.register
class ViaTorch(Benchmark):
    @staticmethod
    def save(t: torch.Tensor) -> bytes:
        return pickle.dumps(
            (memoryview(t.numpy()).tobytes(), t.shape, t.dtype)
        )

    @staticmethod
    def load(b: bytes) -> torch.Tensor:
        data, shape, dtype = pickle.loads(b)
        return torch.frombuffer(data, dtype=dtype).clone().view(shape)


def main():
    arr = torch.rand(64, 256, 1024)
    arr2 = arr.view(128, 128, -1).clone()
    cnt = 100

    for strategy in Benchmark.benchmarks():
        start = time.time()
        for _ in range(cnt):
            arr_ = strategy.save(arr)
        save_time = (time.time() - start) * 1000
        print(f"{strategy.__name__} save {save_time / cnt:.2f} ms/op")

        to_ld_arr = strategy.save(arr2)

        start = time.time()
        for _ in range(cnt):
            ld_arr = strategy.load(to_ld_arr)
        load_time = (time.time() - start) * 1000
        print(f"{strategy.__name__} load {load_time / cnt:.2f} ms/op")

        print(f"{strategy.__name__} size {len(arr_) / 1.e6 :.2f} MB")

        assert (ld_arr == arr2).all(), "Loaded array is not equal to original"
        print("---")


if __name__ == "__main__":
    main()
