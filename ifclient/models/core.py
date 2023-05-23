from typing import Any, Dict, Optional, Type, Callable, Generic, TypeVar
from abc import ABCMeta, abstractmethod
from pydantic import BaseModel, Field

from ..core import HttpClient
from ..core import TritonClient
from ..utils import log_times
from ..utils import log_endpoint


# algorithms
def register_core(
    name: str,
    global_dict: Dict[str, type],
    *,
    before_register: Optional[Callable] = None,
    after_register: Optional[Callable] = None,
):
    def _register(cls):
        if before_register is not None:
            before_register(cls)
        registered = global_dict.get(name)
        if registered is not None:
            print(
                f"> [warning] '{name}' has already registered "
                f"in the given global dict ({global_dict})"
            )
            return cls
        global_dict[name] = cls
        if after_register is not None:
            after_register(cls)
        return cls

    return _register


T = TypeVar("T", bound="WithRegister", covariant=True)


class WithRegister(Generic[T]):
    d: Dict[str, Type[T]]
    __identifier__: str

    @classmethod
    def get(cls, name: str) -> Type[T]:
        return cls.d[name]

    @classmethod
    def has(cls, name: str) -> bool:
        return name in cls.d

    @classmethod
    def make(cls, name: str, config: Dict[str, Any]) -> T:
        return cls.get(name)(**config)  # type: ignore

    @classmethod
    def register(cls, name: str) -> Callable[[Type[T]], Type[T]]:
        def before(cls_: Type[T]) -> None:
            cls_.__identifier__ = name

        return register_core(name, cls.d, before_register=before)


algorithms = {}


class AlgorithmBase(WithRegister, metaclass=ABCMeta):
    d = algorithms
    endpoint: str

    def __init__(self, clients: Dict[str, Any]) -> None:
        self.clients = clients
    """
        abstractmethod 是 Python 的 abc 模块中的一个装饰器，用于声明抽象方法。
        抽象方法是在基类中声明但不实现的方法，子类必须实现这些方法。
        如果子类没有实现这些方法，那么在尝试实例化子类时，Python 会抛出 TypeError。
    """
    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    async def run(self, data: Any, *args: Any) -> Any:
        pass

    @property
    def http_client(self) -> Optional[HttpClient]:
        return self.clients.get("http")

    @property
    def triton_client(self) -> Optional[TritonClient]:
        return self.clients.get("triton")

    def log_endpoint(self, data: BaseModel) -> None:
        log_endpoint(self.endpoint, data)

    def log_times(self, latencies: Dict[str, float]) -> None:
        log_times(self.endpoint, latencies)


class TextModel(BaseModel):
    text: str = Field(..., description="The text that we want to handle.")


class ImageModel(BaseModel):
    url: str = Field(
        ...,
        description="""
The `cdn` / `cos` url of the user's image.
> `cos` url from `qcloud` is preferred.
"""
    )


__all__ = [
    "algorithms",
    "AlgorithmBase",
    "TextModel",
    "ImageModel"
]