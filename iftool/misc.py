"""
在编程中，"misc"通常是"miscellaneous"的缩写，用于表示各种各样的、不同种类的，或者是没有明确分类的事物。
当一个文件被命名为"misc.py"时，通常意味着这个文件包含了一些不易于分类，或者与其他文件中的代码不直接相关的代码。
"""
import os
import json
# 用于定义抽象基类，并检查一个类是否满足抽象基类定义的接口
from abc import abstractmethod, ABC
from typing import Dict, Any


# 字典的浅拷贝
def shallow_copy_dict(d: dict) -> dict:
    d = d.copy()
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = shallow_copy_dict(v)
    return d


class OPTBase(ABC):
    def __init__(self):
        self._opt = self.defaults
        self.update_from_env()

    @property
    @abstractmethod
    def env_key(self) -> str:
        pass

    @property
    @abstractmethod
    def defaults(self) -> Dict[str, Any]:
        pass

    def __getattr__(self, __name: str) -> Any:
        return self._opt[__name]

    def update_from_env(self) -> None:
        env_opt_json = os.environ.get(self.env_key)
        if env_opt_json is not None:
            self._opt.update(json.loads(env_opt_json))

    def opt_context(self, increment: Dict[str, Any]) -> Any:
        class _:
            def __init__(self) -> None:
                self._increment = increment
                self._backup = shallow_copy_dict(instance._opt)

            def __enter__(self) -> None:
                instance._opt.update(self._increment)

            def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                instance._opt.update(self._backup)

        instance = self
        return _()

    def opt_env_context(self, increment: Dict[str, Any]) -> Any:
        class _:
            def __init__(self) -> None:
                self._increment = increment
                self._backup = os.environ.get(instance.env_key)

            def __enter__(self) -> None:
                os.environ[instance.env_key] = json.dumps(self._increment)

            def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                if self._backup is None:
                    del os.environ[instance.env_key]
                else:
                    os.environ[instance.env_key] = self._backup

        instance = self
        return _()
