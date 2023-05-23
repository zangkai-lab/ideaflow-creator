"""
在编程中，"misc"通常是"miscellaneous"的缩写，用于表示各种各样的、不同种类的，或者是没有明确分类的事物。
当一个文件被命名为"misc.py"时，通常意味着这个文件包含了一些不易于分类，或者与其他文件中的代码不直接相关的代码。
"""
import os
import json
import dill
import random
import zipfile
import numpy as np
from tqdm import tqdm
from datetime import timedelta
"""
shutil是Python的一个标准库，提供了一些高级文件操作，包括复制、移动、删除文件或目录等
它是"shell utilities"的缩写，这些操作相当于shell编程中的各种操作
"""
import shutil
import time
import math
import logging
import hashlib
import operator
import unicodedata
"""
Python中的decimal模块提供了Decimal数据类型用于浮点数的精确计算
"""
import decimal

"""
errno模块定义了一些与C语言中的errno相对应的符号
这些符号表示各种错误类型。在Unix和其他POSIX兼容的系统中
errno模块包含的这些符号在底层C库中被用于报告系统调用的错误。
"""
import errno
import sys

"""
inspect是Python的内建模块，它提供了很多有用的函数来帮助获取对象的信息，比如模块、类、函数、追踪记录、帧以及协程。
这些信息通常包括类的成员、文档字符串、源代码、规格以及函数的参数等
"""
import inspect

import threading

# 用于定义抽象基类，并检查一个类是否满足抽象基类定义的接口
from abc import abstractmethod, ABC, ABCMeta
"""
Callable 表示可调用的类型，例如函数或者方法
Protocol 是 Python 3.8 引入的新特性，用于结构化类型检查；可以理解为 Protocol 是一种更灵活的接口定义方式
Generic 是一个元类，用于定义用户自定义的泛型类。泛型类是可以接受一个或多个类型参数的类
"""
from typing import Dict, Any, Optional, TypeVar, Callable, Protocol, Generic, Type, List, Union, Iterable, Set
from functools import reduce, partial
from datetime import datetime
from collections import OrderedDict
from dataclasses import dataclass, Field, fields, asdict

from .types import configs_type, np_dict_type
from .constants import TIME_FORMAT


# utils ************************************************************************************************************
class PureFromInfoMixin:
    def from_info(self, info: Dict[str, Any]) -> None:
        for k, v in info.items():
            setattr(self, k, v)


def get_latest_workspace(root: str) -> Optional[str]:
    if not os.path.isdir(root):
        return None
    all_workspaces = []
    for stuff in os.listdir(root):
        if not os.path.isdir(os.path.join(root, stuff)):
            continue
        try:
            datetime.strptime(stuff, TIME_FORMAT)
            all_workspaces.append(stuff)
        except:
            pass
    if not all_workspaces:
        return None
    return os.path.join(root, sorted(all_workspaces)[-1])


def walk(
    root: str,
    hierarchy_callback: Callable[[List[str], str], None],
    filter_extensions: Optional[Set[str]] = None,
) -> None:
    walked = list(os.walk(root))
    for folder, _, files in tqdm(walked, desc="folders", position=0, mininterval=1):
        for file in tqdm(files, desc="files", position=1, leave=False, mininterval=1):
            if filter_extensions is not None:
                if not any(file.endswith(ext) for ext in filter_extensions):
                    continue
            hierarchy = folder.split(os.path.sep) + [file]
            hierarchy_callback(hierarchy, os.path.join(folder, file))


def grouped_into(iterable: Iterable, n: int) -> List[tuple]:
    """Group an iterable into `n` groups."""

    elements = list(iterable)
    num_elements = len(elements)
    num_elem_per_group = int(math.ceil(num_elements / n))
    results: List[tuple] = []
    split_idx = num_elements + n - n * num_elem_per_group
    start = 0
    for i in range(split_idx):
        end = start + num_elem_per_group
        results.append(tuple(elements[start:end]))
        start = end
    for i in range(split_idx, n):
        end = start + num_elem_per_group - 1
        results.append(tuple(elements[start:end]))
        start = end
    return results


class DownloadProgressBar(tqdm):
    def update_to(
        self,
        b: int = 1,
        bsize: int = 1,
        tsize: Optional[int] = None,
    ) -> None:
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def prod(iterable: Iterable) -> float:
    """Return cumulative production of an iterable."""

    return float(reduce(operator.mul, iterable, 1))


def hash_code(code: str) -> str:
    """Return hash code for a string."""

    code = code.encode()
    return hashlib.md5(code).hexdigest()


def is_numeric(s: Any) -> bool:
    """Check whether `s` is a number."""

    try:
        s = float(s)
        return True
    except (TypeError, ValueError):
        try:
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            return False


def hash_dict(d: Dict[str, Any]) -> str:
    """Return a consistent hash code for an arbitrary dict."""

    def _hash(_d: Dict[str, Any]) -> str:
        sorted_keys = sorted(_d)
        hashes = []
        for k in sorted_keys:
            v = _d[k]
            if isinstance(v, dict):
                hashes.append(_hash(v))
            elif isinstance(v, set):
                hashes.append(hash_code(str(sorted(v))))
            else:
                hashes.append(hash_code(str(v)))
        return hash_code("".join(hashes))

    return _hash(d)


def random_hash() -> str:
    return hash_code(str(random.random()))


def get_requirements(fn: Any) -> List[str]:
    if isinstance(fn, type):
        fn = fn.__init__  # type: ignore
    requirements = []
    signature = inspect.signature(fn)
    for k, param in signature.parameters.items():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        if param.default is not inspect.Parameter.empty:
            continue
        requirements.append(k)
    return requirements


def get_arguments(
    *,
    num_back: int = 0,
    pop_class_attributes: bool = True,
) -> Dict[str, Any]:
    frame = inspect.currentframe().f_back  # type: ignore
    for _ in range(num_back):
        frame = frame.f_back
    if frame is None:
        raise ValueError("`get_arguments` should be called inside a frame")
    arguments = inspect.getargvalues(frame)[-1]
    if pop_class_attributes:
        arguments.pop("self", None)
        arguments.pop("__class__", None)
    return arguments


def prepare_workspace_from(
    workspace: str,
    *,
    timeout: timedelta = timedelta(30),
    make: bool = True,
) -> str:
    current_time = datetime.now()
    if os.path.isdir(workspace):
        for stuff in os.listdir(workspace):
            if not os.path.isdir(os.path.join(workspace, stuff)):
                continue
            try:
                stuff_time = datetime.strptime(stuff, TIME_FORMAT)
                stuff_delta = current_time - stuff_time
                if stuff_delta > timeout:
                    msg = f"{stuff} will be removed (already {stuff_delta} ago)"
                    print_warning(msg)
                    shutil.rmtree(os.path.join(workspace, stuff))
            except:
                pass
    workspace = os.path.join(workspace, current_time.strftime(TIME_FORMAT))
    if make:
        os.makedirs(workspace)
    return workspace


def get_num_positional_args(fn: Callable) -> Union[int, float]:
    signature = inspect.signature(fn)
    counter = 0
    for param in signature.parameters.values():
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            return math.inf
        if param.kind is inspect.Parameter.POSITIONAL_ONLY:
            counter += 1
        elif param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
            counter += 1
    return counter


class IWithRequirements:
    @classmethod
    def requirements(cls) -> List[str]:
        requirements = get_requirements(cls)
        requirements.remove("self")
        return requirements


# 序列化数据 **********************************************************************************************************
TRegister = TypeVar("TRegister", bound="WithRegister", covariant=True)
TSerializable = TypeVar("TSerializable", bound="ISerializable", covariant=True)
TSerializableArrays = TypeVar(
    "TSerializableArrays",
    bound="ISerializableArrays",
    covariant=True,
)
TSArrays = TypeVar("TSArrays", bound="ISerializableArrays", covariant=True)
TSDataClass = TypeVar("TSDataClass", bound="ISerializableDataClass", covariant=True)
TDataClass = TypeVar("TDataClass", bound="DataClassBase")


def register_core(
    name: str,
    global_dict: Dict[str, type],
    *,
    allow_duplicate: bool = False,
    before_register: Optional[Callable] = None,
    after_register: Optional[Callable] = None,
):
    def _register(cls):
        if before_register is not None:
            before_register(cls)
        registered = global_dict.get(name)
        if registered is not None and not allow_duplicate:
            print_warning(
                f"'{name}' has already registered "
                f"in the given global dict ({global_dict})"
            )
            return cls
        global_dict[name] = cls
        if after_register is not None:
            after_register(cls)
        return cls

    return _register


class WithRegister(Generic[TRegister]):
    d: Dict[str, Type[TRegister]]
    __identifier__: str

    @classmethod
    def get(cls: Type[TRegister], name: str) -> Type[TRegister]:
        return cls.d[name]

    @classmethod
    def has(cls, name: str) -> bool:
        return name in cls.d

    @classmethod
    def make(
        cls: Type[TRegister],
        name: str,
        config: Dict[str, Any],
        *,
        ensure_safe: bool = False,
    ) -> TRegister:
        base = cls.get(name)
        if not ensure_safe:
            return base(**config)  # type: ignore
        return safe_execute(base, config)

    @classmethod
    def make_multiple(
        cls: Type[TRegister],
        names: Union[str, List[str]],
        configs: configs_type = None,
        *,
        ensure_safe: bool = False,
    ) -> List[TRegister]:
        if configs is None:
            configs = {}
        if isinstance(names, str):
            assert isinstance(configs, dict)
            return cls.make(names, configs, ensure_safe=ensure_safe)  # type: ignore
        if not isinstance(configs, list):
            configs = [configs.get(name, {}) for name in names]
        return [
            cls.make(name, shallow_copy_dict(config), ensure_safe=ensure_safe)
            for name, config in zip(names, configs)
        ]

    @classmethod
    def register(
        cls,
        name: str,
        *,
        allow_duplicate: bool = False,
    ) -> Callable:
        def before(cls_: Type) -> None:
            cls_.__identifier__ = name

        return register_core(
            name,
            cls.d,
            allow_duplicate=allow_duplicate,
            before_register=before,
        )

    @classmethod
    def remove(cls, name: str) -> Callable:
        return cls.d.pop(name)

    @classmethod
    def check_subclass(cls, name: str) -> bool:
        return issubclass(cls.d[name], cls)


@dataclass
class DataClassBase(ABC):
    @property
    def fields(self) -> List[Field]:
        return fields(self)

    @property
    def field_names(self) -> List[str]:
        return [f.name for f in self.fields]

    @property
    def attributes(self) -> List[Any]:
        return [getattr(self, name) for name in self.field_names]

    def asdict(self) -> Dict[str, Any]:
        return asdict(self)

    def copy(self: TDataClass) -> TDataClass:
        return type(self)(**shallow_copy_dict(asdict(self)))

    def update_with(self: TDataClass, other: TDataClass) -> TDataClass:
        d = update_dict(other.asdict(), self.asdict())
        return self.__class__(**d)


@dataclass
class JsonPack(DataClassBase):
    type: str
    info: Dict[str, Any]


class ISerializable(WithRegister, Generic[TSerializable], metaclass=ABCMeta):
    # abstract

    @abstractmethod
    def to_info(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def from_info(self, info: Dict[str, Any]) -> None:
        pass

    # optional callbacks

    def after_load(self) -> None:
        pass

    # api

    def to_pack(self) -> JsonPack:
        return JsonPack(self.__identifier__, self.to_info())

    @classmethod
    def from_pack(cls: Type[TSerializable], pack: Dict[str, Any]) -> TSerializable:
        obj: ISerializable = cls.make(pack["type"], {})
        obj.from_info(pack["info"])
        obj.after_load()
        return obj

    def to_json(self) -> str:
        return json.dumps(self.to_pack().asdict())

    @classmethod
    def from_json(cls: Type[TSerializable], json_string: str) -> TSerializable:
        return cls.from_pack(json.loads(json_string))

    def copy(self: TSerializable) -> TSerializable:
        copied = self.__class__()
        copied.from_info(shallow_copy_dict(self.to_info()))
        return copied


@dataclass
class ISerializableDataClass(
    ISerializable,
    DataClassBase,
    Generic[TSDataClass],
    metaclass=ABCMeta,
):
    @classmethod
    @abstractmethod
    def d(cls) -> Dict[str, Type["ISerializableDataClass"]]:
        pass

    def to_info(self) -> Dict[str, Any]:
        return self.asdict()

    def from_info(self, info: Dict[str, Any]) -> None:
        for k, v in info.items():
            setattr(self, k, v)

    @classmethod
    def get(cls: Type[TRegister], name: str) -> Type[TRegister]:
        return cls.d()[name]

    @classmethod
    def has(cls, name: str) -> bool:
        return name in cls.d()

    @classmethod
    def register(
        cls,
        name: str,
        *,
        allow_duplicate: bool = False,
    ) -> Callable:
        def before(cls_: Type) -> None:
            cls_.__identifier__ = name

        return register_core(
            name,
            cls.d(),
            allow_duplicate=allow_duplicate,
            before_register=before,
        )

    @classmethod
    def remove(cls, name: str) -> Callable:
        return cls.d().pop(name)

    @classmethod
    def check_subclass(cls, name: str) -> bool:
        return issubclass(cls.d()[name], cls)


class ISerializableArrays(ISerializable, Generic[TSArrays], metaclass=ABCMeta):
    @abstractmethod
    def to_npd(self) -> np_dict_type:
        pass

    @abstractmethod
    def from_npd(self, npd: np_dict_type) -> None:
        pass

    def copy(self: TSerializableArrays) -> TSerializableArrays:
        copied: TSerializableArrays = super().copy()
        copied.from_npd(shallow_copy_dict(self.to_npd()))
        return copied


class Serializer:
    id_file: str = "id.txt"
    info_file: str = "info.json"
    npd_folder: str = "npd"

    @classmethod
    def save_info(
        cls,
        folder: str,
        *,
        info: Optional[Dict[str, Any]] = None,
        serializable: Optional[ISerializable] = None,
    ) -> None:
        os.makedirs(folder, exist_ok=True)
        if info is None and serializable is None:
            raise ValueError("either `info` or `serializable` should be provided")
        if info is None:
            info = serializable.to_info()
        with open(os.path.join(folder, cls.info_file), "w") as f:
            json.dump(info, f)

    @classmethod
    def load_info(cls, folder: str) -> Dict[str, Any]:
        return cls.try_load_info(folder, strict=True)

    @classmethod
    def try_load_info(
        cls,
        folder: str,
        *,
        strict: bool = False,
    ) -> Optional[Dict[str, Any]]:
        info_path = os.path.join(folder, cls.info_file)
        if not os.path.isfile(info_path):
            if not strict:
                return
            raise ValueError(f"'{info_path}' does not exist")
        with open(info_path, "r") as f:
            info = json.load(f)
        return info

    @classmethod
    def save_npd(
        cls,
        folder: str,
        *,
        npd: Optional[np_dict_type] = None,
        serializable: Optional[ISerializableArrays] = None,
    ) -> None:
        os.makedirs(folder, exist_ok=True)
        if npd is None and serializable is None:
            raise ValueError("either `npd` or `serializable` should be provided")
        if npd is None:
            npd = serializable.to_npd()
        npd_folder = os.path.join(folder, cls.npd_folder)
        os.makedirs(npd_folder, exist_ok=True)
        for k, v in npd.items():
            np.save(os.path.join(npd_folder, f"{k}.npy"), v)

    @classmethod
    def load_npd(cls, folder: str) -> np_dict_type:
        os.makedirs(folder, exist_ok=True)
        npd_folder = os.path.join(folder, cls.npd_folder)
        if not os.path.isdir(npd_folder):
            raise ValueError(f"'{npd_folder}' does not exist")
        npd = {}
        for file in os.listdir(npd_folder):
            key = os.path.splitext(file)[0]
            npd[key] = np.load(os.path.join(npd_folder, file))
        return npd

    @classmethod
    def save(
        cls,
        folder: str,
        serializable: ISerializable,
        *,
        save_npd: bool = True,
    ) -> None:
        cls.save_info(folder, serializable=serializable)
        if save_npd and isinstance(serializable, ISerializableArrays):
            cls.save_npd(folder, serializable=serializable)
        with open(os.path.join(folder, cls.id_file), "w") as f:
            f.write(serializable.__identifier__)

    @classmethod
    def load(
        cls,
        folder: str,
        base: Type[TSerializable],
        *,
        swap_id: Optional[str] = None,
        swap_info: Optional[Dict[str, Any]] = None,
        load_npd: bool = True,
    ) -> TSerializable:
        serializable = cls.load_empty(folder, base, swap_id=swap_id)
        serializable.from_info(swap_info or cls.load_info(folder))
        if load_npd and isinstance(serializable, ISerializableArrays):
            serializable.from_npd(cls.load_npd(folder))
        return serializable

    @classmethod
    def load_empty(
        cls,
        folder: str,
        base: Type[TSerializable],
        *,
        swap_id: Optional[str] = None,
    ) -> TSerializable:
        if swap_id is not None:
            s_type = swap_id
        else:
            id_path = os.path.join(folder, cls.id_file)
            if not os.path.isfile(id_path):
                raise ValueError(f"cannot find '{id_path}'")
            with open(id_path, "r") as f:
                s_type = f.read().strip()
        return base.make(s_type, {})



# 小工具 **********************************************************************************************************
# 字典的浅拷贝
def shallow_copy_dict(d: dict) -> dict:
    d = d.copy()
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = shallow_copy_dict(v)
    return d


def sort_dict_by_value(d: Dict[Any, Any], *, reverse: bool = False) -> OrderedDict:
    sorted_items = sorted([(v, k) for k, v in d.items()], reverse=reverse)
    return OrderedDict({item[1]: item[0] for item in sorted_items})


def fix_float_to_length(num: float, length: int) -> str:
    """Change a float number to string format with fixed length."""

    ctx = decimal.Context()
    ctx.prec = 2 * length
    d = ctx.create_decimal(repr(num))
    str_num = format(d, "f").lower()
    if str_num == "nan":
        return f"{str_num:^{length}s}"
    idx = str_num.find(".")
    if idx == -1:
        diff = length - len(str_num)
        if diff <= 0:
            return str_num
        if diff == 1:
            return f"{str_num}."
        return f"{str_num}.{'0' * (diff - 1)}"
    length = max(length, idx)
    return str_num[:length].ljust(length, "0")


def update_dict(src_dict: dict, tgt_dict: dict) -> dict:
    """
    Update tgt_dict with src_dict.
    * Notice that changes will happen only on keys which src_dict holds.

    Parameters
    ----------
    src_dict : dict
    tgt_dict : dict

    Returns
    -------
    tgt_dict : dict

    """

    for k, v in src_dict.items():
        tgt_v = tgt_dict.get(k)
        if tgt_v is None:
            tgt_dict[k] = v
        elif not isinstance(v, dict):
            tgt_dict[k] = v
        else:
            update_dict(v, tgt_v)
    return tgt_dict


# Fn  ***************************************************************************************************************
TFnResponse = TypeVar("TFnResponse")


def check_requires(fn: Any, name: str, strict: bool = True) -> bool:
    if isinstance(fn, type):
        fn = fn.__init__  # type: ignore
    signature = inspect.signature(fn)
    for k, param in signature.parameters.items():
        if not strict and param.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if k == name:
            if param.kind is inspect.Parameter.VAR_POSITIONAL:
                return False
            return True
    return False


def filter_kw(
    fn: Callable,
    kwargs: Dict[str, Any],
    *,
    strict: bool = False,
) -> Dict[str, Any]:
    kw = {}
    for k, v in kwargs.items():
        if check_requires(fn, k, strict):
            kw[k] = v
    return kw


class Fn(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> TFnResponse:
        pass


def safe_execute(fn: Fn, kw: Dict[str, Any], *, strict: bool = False) -> TFnResponse:
    return fn(**filter_kw(fn, kw, strict=strict))


# Logging  ********************************************************************************************************
class Incrementer:
    """
    Util class which can calculate running mean & running std efficiently.

    Parameters
    ----------
    window_size : {int, None}, window size of running statistics.
    * If None, then all history records will be used for calculation.
    """

    def __init__(self, window_size: int = None):
        if window_size is not None:
            if not isinstance(window_size, int):
                msg = f"window size should be integer, {type(window_size)} found"
                raise ValueError(msg)
            if window_size < 2:
                msg = f"window size should be greater than 2, {window_size} found"
                raise ValueError(msg)
        self._window_size = window_size
        self._n_record = self._previous = None
        self._running_sum = self._running_square_sum = None

    @property
    def mean(self):
        return self._running_sum / self._n_record

    @property
    def std(self):
        return math.sqrt(
            max(
                0.0,
                self._running_square_sum / self._n_record - self.mean ** 2,
            )
        )

    @property
    def n_record(self):
        return self._n_record

    def update(self, new_value):
        if self._n_record is None:
            self._n_record = 1
            self._running_sum = new_value
            self._running_square_sum = new_value ** 2
        else:
            self._n_record += 1
            self._running_sum += new_value
            self._running_square_sum += new_value ** 2
        if self._window_size is not None:
            if self._previous is None:
                self._previous = [new_value]
            else:
                self._previous.append(new_value)
            if self._n_record == self._window_size + 1:
                self._n_record -= 1
                previous = self._previous.pop(0)
                self._running_sum -= previous
                self._running_square_sum -= previous ** 2


class _Formatter(logging.Formatter):
    """Formatter for logging, which supports millisecond."""

    converter = datetime.fromtimestamp

    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            s = ct.strftime(datefmt)
        else:
            t = ct.strftime("%Y-%m-%d %H:%M:%S")
            s = "%s.%03d" % (t, record.msecs)
        return s

    def formatMessage(self, record: logging.LogRecord) -> str:
        record.__dict__.setdefault("func_prefix", "Unknown")
        return super().formatMessage(record)


def truncate_string_to_length(string: str, length: int) -> str:
    """Truncate a string to make sure its length not exceeding a given length."""

    if len(string) <= length:
        return string
    half_length = int(0.5 * length) - 1
    head = string[:half_length]
    tail = string[-half_length:]
    return f"{head}{'.' * (length - 2 * half_length)}{tail}"


def timestamp(simplify: bool = False, ensure_different: bool = False) -> str:
    """
    Return current timestamp.

    Parameters
    ----------
    simplify : bool. If True, format will be simplified to 'year-month-day'.
    ensure_different : bool. If True, format will include millisecond.

    Returns
    -------
    timestamp : str

    """

    now = datetime.now()
    if simplify:
        return now.strftime("%Y-%m-%d")
    if ensure_different:
        return now.strftime("%Y-%m-%d_%H-%M-%S-%f")
    return now.strftime("%Y-%m-%d_%H-%M-%S")


class LoggingMixin:
    """
    Mixin class to provide logging methods for base class.

    Attributes
    ----------
    _triggered_ : bool
    * If not `_triggered_`, log file will not be created.

    _verbose_level_ : int
    * Preset verbose level of the whole logging process.

    Methods
    ----------
    log_msg(self, body, prefix="", verbose_level=1)
        Log something either through console or to a file.
        * body : str
            Main logging message.
        * prefix : str
            Prefix added to `body` when logging message goes through console.
        * verbose_level : int
            If `self._verbose_level_` >= verbose_level, then logging message
            will go through console.

    log_block_msg(self, body, prefix="", title="", verbose_level=1)
        Almost the same as `log_msg`, except adding `title` on top of `body`.

    """

    _triggered_ = False
    _initialized_ = False
    _logging_path_ = None
    _logger_ = _verbose_level_ = None
    _date_format_string_ = "%Y-%m-%d %H:%M:%S.%f"
    _formatter_ = _Formatter(
        "[ {asctime:s} ] [ {levelname:^8s} ] {func_prefix:s} {message:s}",
        _date_format_string_,
        style="{",
    )
    _timing_dict_, _time_cache_dict_ = {}, {}

    info_prefix = ">  [ info ] "
    warning_prefix = "> [warning] "
    error_prefix = "> [ error ] "

    @property
    def logging_path(self):
        if self._logging_path_ is None:
            folder = os.path.join(os.getcwd(), "_logging", type(self).__name__)
            os.makedirs(folder, exist_ok=True)
            self._logging_path_ = self.generate_logging_path(folder)
        return self._logging_path_

    @property
    def console_handler(self):
        if self._logger_ is None:
            return
        for handler in self._logger_.handlers:
            if isinstance(handler, logging.StreamHandler):
                return handler

    @staticmethod
    def _get_func_prefix(frame=None, return_prefix=True):
        if frame is None:
            frame = inspect.currentframe().f_back.f_back
        if not return_prefix:
            return frame
        frame_info = inspect.getframeinfo(frame)
        file_name = truncate_string_to_length(os.path.basename(frame_info.filename), 16)
        func_name = truncate_string_to_length(frame_info.function, 24)
        func_prefix = (
            f"[ {func_name:^24s} ] [ {file_name:>16s}:{frame_info.lineno:<4d} ]"
        )
        return func_prefix

    @staticmethod
    def _release_handlers(logger):
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    @staticmethod
    def generate_logging_path(folder: str) -> str:
        return os.path.join(folder, f"{timestamp()}.log")

    def _init_logging(self, verbose_level: Optional[int] = 2, trigger: bool = True):
        wants_trigger = trigger and not LoggingMixin._triggered_
        if LoggingMixin._initialized_ and not wants_trigger:
            return self
        LoggingMixin._initialized_ = True
        logger_name = getattr(self, "_logger_name_", "root")
        logger = LoggingMixin._logger_ = logging.getLogger(logger_name)
        LoggingMixin._verbose_level_ = verbose_level
        if not trigger:
            return self
        LoggingMixin._triggered_ = True
        config = getattr(self, "config", {})
        self._logging_path_ = config.get("_logging_path_")
        if self._logging_path_ is None:
            self._logging_path_ = config["_logging_path_"] = self.logging_path
        os.makedirs(os.path.dirname(self.logging_path), exist_ok=True)
        file_handler = logging.FileHandler(self.logging_path, encoding="utf-8")
        file_handler.setFormatter(self._formatter_)
        file_handler.setLevel(logging.DEBUG)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(_Formatter("{custom_prefix:s}{message:s}", style="{"))
        logger.setLevel(logging.DEBUG)
        self._release_handlers(logger)
        logger.addHandler(console)
        logger.addHandler(file_handler)
        self.log_block_msg(sys.version, title="system version", verbose_level=None)
        return self

    def log_msg(
            self,
            body: str,
            prefix: str = "",
            verbose_level: Optional[int] = 1,
            msg_level: int = logging.INFO,
            frame=None,
    ):
        preset_verbose_level = getattr(self, "_verbose_level", None)
        if preset_verbose_level is not None:
            self._verbose_level_ = preset_verbose_level
        elif self._verbose_level_ is None:
            self._verbose_level_ = 0
        console_handler = self.console_handler
        if verbose_level is None or self._verbose_level_ < verbose_level:
            do_print, console_level = False, msg_level + 10
        else:
            do_print, console_level = not LoggingMixin._triggered_, msg_level
        if console_handler is not None:
            console_handler.setLevel(console_level)
        if do_print:
            print(prefix + body)
        elif LoggingMixin._triggered_:
            func_prefix = self._get_func_prefix(frame)
            self._logger_.log(
                msg_level,
                body,
                extra={"func_prefix": func_prefix, "custom_prefix": prefix},
            )
        if console_handler is not None:
            console_handler.setLevel(logging.INFO)

    def log_block_msg(
            self,
            body: str,
            prefix: str = "",
            title: str = "",
            verbose_level: Optional[int] = 1,
            msg_level: int = logging.INFO,
            frame=None,
    ):
        frame = self._get_func_prefix(frame, False)
        self.log_msg(f"{title}\n{body}\n", prefix, verbose_level, msg_level, frame)

    def exception(self, body, frame=None):
        self._logger_.exception(
            body,
            extra={
                "custom_prefix": self.error_prefix,
                "func_prefix": LoggingMixin._get_func_prefix(frame),
            },
        )

    @staticmethod
    def log_with_external_method(body, prefix, log_method, *args, **kwargs):
        if log_method is None:
            print(prefix + body)
        else:
            kwargs["frame"] = LoggingMixin._get_func_prefix(
                kwargs.pop("frame", None),
                False,
            )
            log_method(body, prefix, *args, **kwargs)

    @staticmethod
    def merge_logs_by_time(*log_files, tgt_file):
        tgt_folder = os.path.dirname(tgt_file)
        date_str_len = (
                len(datetime.today().strftime(LoggingMixin._date_format_string_)) + 4
        )
        with lock_manager(tgt_folder, [tgt_file], clear_stuffs_after_exc=False):
            msg_dict, msg_block, last_searched = {}, [], None
            for log_file in log_files:
                with open(log_file, "r") as f:
                    for line in f:
                        date_str = line[:date_str_len]
                        if date_str[:2] == "[ " and date_str[-2:] == " ]":
                            searched_time = datetime.strptime(
                                date_str[2:-2],
                                LoggingMixin._date_format_string_,
                            )
                        else:
                            msg_block.append(line)
                            continue
                        if last_searched is not None:
                            msg_block_ = "".join(msg_block)
                            msg_dict.setdefault(last_searched, []).append(msg_block_)
                        last_searched = searched_time
                        msg_block = [line]
                    if msg_block:
                        msg_dict.setdefault(last_searched, []).append(
                            "".join(msg_block)
                        )
            with open(tgt_file, "w") as f:
                f.write("".join(["".join(msg_dict[key]) for key in sorted(msg_dict)]))

    @classmethod
    def reset_logging(cls) -> None:
        cls._triggered_ = False
        cls._initialized_ = False
        cls._logging_path_ = None
        if cls._logger_ is not None:
            cls._release_handlers(cls._logger_)
        cls._logger_ = cls._verbose_level_ = None
        cls._timing_dict_, cls._time_cache_dict_ = {}, {}

    @classmethod
    def start_timer(cls, name):
        if name in cls._time_cache_dict_:
            print_warning(
                f"'{name}' was already in time cache dict, "
                "this may cause by calling `start_timer` repeatedly"
            )
            return
        cls._time_cache_dict_[name] = time.time()

    @classmethod
    def end_timer(cls, name):
        start_time = cls._time_cache_dict_.pop(name, None)
        if start_time is None:
            print_warning(
                f"'{name}' was not found in time cache dict, "
                "this may cause by not calling `start_timer` method"
            )
            return
        incrementer = cls._timing_dict_.setdefault(name, Incrementer())
        incrementer.update(time.time() - start_time)

    def log_timing(self):
        timing_str_list = ["=" * 138]
        for name in sorted(self._timing_dict_.keys()):
            incrementer = self._timing_dict_[name]
            timing_str_list.append(
                f"|   {name:<82s}   | "
                f"{fix_float_to_length(incrementer.mean, 10)} ± "
                f"{fix_float_to_length(incrementer.std, 10)} | "
                f"{incrementer.n_record:>12d} hits   |"
            )
            timing_str_list.append("-" * 138)
        self.log_block_msg(
            "\n".join(timing_str_list),
            title="timing",
            verbose_level=None,
            msg_level=logging.DEBUG,
        )
        return self


class PureLoggingMixin:
    """
    Mixin class to provide (pure) logging method for base class.

    Attributes
    ----------
    _loggers_ : dict(int, logging.Logger)
        Recorded all loggers initialized.

    _formatter_ : _Formatter
        Formatter for all loggers.

    Methods
    ----------
    log_msg(self, name, msg, msg_level=logging.INFO)
        Log something to a file, with logger initialized by `name`.

    log_block_msg(self, name, title, body, msg_level=logging.INFO)
        Almost the same as `log_msg`, except adding `title` on top of `body`.

    """

    _name = _meta_name = None

    _formatter_ = LoggingMixin._formatter_
    _loggers_: Dict[str, logging.Logger] = {}
    _logger_paths_: Dict[str, str] = {}
    _timing_dict_ = {}

    @property
    def meta_suffix(self):
        return "" if self._meta_name is None else self._meta_name

    @property
    def name_suffix(self):
        return "" if self._name is None else f"-{self._name}"

    @property
    def meta_log_name(self):
        return f"__meta__{self.meta_suffix}{self.name_suffix}"

    @staticmethod
    def get_logging_path(logger):
        logging_path = None
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                logging_path = handler.baseFilename
                break
        if logging_path is None:
            raise ValueError(f"No FileHandler was found in given logger '{logger}'")
        return logging_path

    def _get_logger_info(self, name):
        logger = name if isinstance(name, logging.Logger) else self._loggers_.get(name)
        if logger is None:
            raise ValueError(
                f"logger for '{name}' is not defined, "
                "please call `_setup_logger` first"
            )
        if isinstance(name, str):
            logging_path = self._logger_paths_[name]
        else:
            logging_path = self.get_logging_path(logger)
        return logger, os.path.dirname(logging_path), logging_path

    def _setup_logger(self, name, logging_path, level=logging.DEBUG):
        if name in self._loggers_:
            return
        console = logging.StreamHandler()
        console.setLevel(logging.CRITICAL)
        console.setFormatter(self._formatter_)
        file_handler = logging.FileHandler(logging_path)
        file_handler.setFormatter(self._formatter_)
        file_handler.setLevel(level)
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        LoggingMixin._release_handlers(logger)
        logger.addHandler(console)
        logger.addHandler(file_handler)
        PureLoggingMixin._loggers_[name] = logger
        PureLoggingMixin._logger_paths_[name] = logging_path
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.CRITICAL)
        self.log_block_msg(name, "system version", sys.version)

    def _log_meta_msg(self, msg, msg_level=logging.INFO, frame=None):
        if frame is None:
            frame = inspect.currentframe().f_back
        self.log_msg(self.meta_log_name, msg, msg_level, frame)

    def _log_with_meta(self, task_name, msg, msg_level=logging.INFO, frame=None):
        if frame is None:
            frame = inspect.currentframe().f_back
        self._log_meta_msg(f"{task_name} {msg}", msg_level, frame)
        self.log_msg(task_name, f"current task {msg}", msg_level, frame)

    def log_msg(self, name, msg, msg_level=logging.INFO, frame=None):
        logger, logging_folder, logging_path = self._get_logger_info(name)
        with lock_manager(
            logging_folder,
            [logging_path],
            clear_stuffs_after_exc=False,
        ):
            logger.log(
                msg_level,
                msg,
                extra={
                    "custom_prefix": "",
                    "func_prefix": LoggingMixin._get_func_prefix(frame),
                },
            )
        return logger

    def log_block_msg(self, name, title, body, msg_level=logging.INFO, frame=None):
        frame = LoggingMixin._get_func_prefix(frame, False)
        self.log_msg(name, f"{title}\n{body}\n", msg_level, frame)

    def exception(self, name, msg, frame=None):
        logger, logging_folder, logging_path = self._get_logger_info(name)
        with lock_manager(
            logging_folder,
            [logging_path],
            clear_stuffs_after_exc=False,
        ):
            logger.exception(
                msg,
                extra={
                    "custom_prefix": LoggingMixin.error_prefix,
                    "func_prefix": LoggingMixin._get_func_prefix(frame),
                },
            )

    def del_logger(self, name):
        logger = self.log_msg(name, f"clearing up logger information of '{name}'")
        del self._loggers_[name], self._logger_paths_[name]
        LoggingMixin._release_handlers(logger)
        del logger


"""
定期刷新一个锁文件
"""


class _lock_file_refresher(threading.Thread):
    def __init__(self, lock_file, delay=1, refresh=0.01):
        super().__init__()
        self.__stop_event = threading.Event()
        self._lock_file, self._delay, self._refresh = lock_file, delay, refresh
        with open(lock_file, "r") as f:
            self._lock_file_contents = f.read()

    def run(self) -> None:
        counter = 0
        while True:
            counter += 1
            time.sleep(self._refresh)
            if counter * self._refresh >= self._delay:
                counter = 0
                with open(self._lock_file, "w") as f:
                    prefix = "\n\n"
                    add_line = f"{prefix}refreshed at {timestamp()}"
                    f.write(self._lock_file_contents + add_line)
            if self.__stop_event.is_set():
                break

    def stop(self):
        self.__stop_event.set()


class context_error_handler:
    """Util class which provides exception handling when using context manager."""

    @property
    def exception_suffix(self):
        return ""

    def _normal_exit(self, exc_type, exc_val, exc_tb):
        pass

    def _exception_exit(self, exc_type, exc_val, exc_tb):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_type:
            self._normal_exit(exc_type, exc_val, exc_tb)
        else:
            self._exception_exit(exc_type, exc_val, exc_tb)


class lock_manager(context_error_handler, LoggingMixin):
    """
    Util class to make simultaneously-write process safe with some
    hacked (ugly) tricks.
    """

    delay = 0.01
    __lock__ = "__lock__"

    def __init__(
            self,
            workspace,
            stuffs,
            verbose_level=None,
            set_lock=True,
            clear_stuffs_after_exc=True,
            name=None,
            wait=1000,
    ):
        self._workspace = workspace
        self._verbose_level = verbose_level
        self._name, self._wait = name, wait
        os.makedirs(workspace, exist_ok=True)
        self._stuffs, self._set_lock = stuffs, set_lock
        self._clear_stuffs = clear_stuffs_after_exc
        self._is_locked = False

    def __enter__(self):
        frame = inspect.currentframe().f_back
        self.log_msg(
            f"waiting for lock at {self.lock_file}",
            self.info_prefix,
            5,
            logging.DEBUG,
            frame,
        )
        enter_time = file_modify = None
        while True:
            try:
                fd = os.open(self.lock_file, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                self.log_msg(
                    "lock acquired",
                    self.info_prefix,
                    5,
                    logging.DEBUG,
                    frame,
                )
                if not self._set_lock:
                    self.log_msg(
                        "releasing lock since set_lock=False",
                        self.info_prefix,
                        5,
                        logging.DEBUG,
                        frame,
                    )
                    os.unlink(self.lock_file)
                    self.__refresher = None
                else:
                    self.log_msg(
                        "writing info to lock file",
                        self.info_prefix,
                        5,
                        logging.DEBUG,
                        frame,
                    )
                    with os.fdopen(fd, "a") as f:
                        f.write(
                            f"name      : {self._name}\n"
                            f"timestamp : {timestamp()}\n"
                            f"workspace : {self._workspace}\n"
                            f"stuffs    :\n{self.cache_stuffs_str}"
                        )
                    self.__refresher = _lock_file_refresher(self.lock_file)
                    self.__refresher.start()
                break
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
                try:
                    if file_modify is None:
                        enter_time = time.time()
                        file_modify = os.path.getmtime(self.lock_file)
                    else:
                        new_file_modify = os.path.getmtime(self.lock_file)
                        if new_file_modify != file_modify:
                            enter_time = time.time()
                            file_modify = new_file_modify
                        else:
                            wait_time = time.time() - enter_time
                            if wait_time >= self._wait:
                                raise ValueError(
                                    f"'{self.lock_file}' has been waited "
                                    f"for too long ({wait_time})"
                                )
                    time.sleep(random.random() * self.delay + self.delay)
                except ValueError:
                    msg = f"lock_manager was blocked by dead lock ({self.lock_file})"
                    self.exception(msg)
                    raise
                except FileNotFoundError:
                    pass
        self.log_block_msg(
            self.cache_stuffs_str,
            title="start processing following stuffs:",
            verbose_level=5,
            msg_level=logging.DEBUG,
            frame=frame,
        )
        self._is_locked = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.__refresher is not None:
            self.__refresher.stop()
            self.__refresher.join()
        if self._set_lock:
            super().__exit__(exc_type, exc_val, exc_tb)

    def _normal_exit(self, exc_type, exc_val, exc_tb, frame=None):
        if self._set_lock:
            os.unlink(self.lock_file)
        if frame is None:
            frame = inspect.currentframe().f_back.f_back.f_back
        self.log_msg("lock released", self.info_prefix, 5, logging.DEBUG, frame)

    def _exception_exit(self, exc_type, exc_val, exc_tb):
        frame = inspect.currentframe().f_back.f_back.f_back
        if self._clear_stuffs:
            for stuff in self._stuffs:
                if os.path.isfile(stuff):
                    self.log_msg(
                        f"clearing cached file: {stuff}",
                        ">> ",
                        5,
                        logging.ERROR,
                        frame,
                    )
                    os.remove(stuff)
                elif os.path.isdir(stuff):
                    self.log_msg(
                        f"clearing cached directory: {stuff}",
                        ">> ",
                        5,
                        logging.ERROR,
                        frame,
                    )
                    shutil.rmtree(stuff)
        self._normal_exit(exc_type, exc_val, exc_tb, frame)

    @property
    def locked(self):
        return self._is_locked

    @property
    def available(self):
        return not os.path.isfile(self.lock_file)

    @property
    def cache_stuffs_str(self):
        return "\n".join([f">> {stuff}" for stuff in self._stuffs])

    @property
    def exception_suffix(self):
        return f", clearing caches for safety{self.logging_suffix}"

    @property
    def lock_file(self):
        return os.path.join(self._workspace, self.__lock__)

    @property
    def logging_suffix(self):
        return "" if self._name is None else f" - {self._name}"


def print_info(msg: str) -> None:
    print(f"{LoggingMixin.info_prefix}{msg}")


def print_warning(msg: str) -> None:
    print(f"{LoggingMixin.warning_prefix}{msg}")


def print_error(msg: str) -> None:
    print(f"{LoggingMixin.error_prefix}{msg}")


def get_err_msg(err: Exception) -> str:
    return " | ".join(map(repr, sys.exc_info()[:2] + (str(err),)))


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


# saving **************************************************************
class Saving(LoggingMixin):
    """
    Util class for saving instances.

    Methods
    ----------
    save_instance(instance, folder, log_method=None)
        Save instance to `folder`.
        * instance : object, instance to save.
        * folder : str, folder to save to.
        * log_method : {None, function}, used as `log_method` parameter in
        `log_with_external_method` method of `LoggingMixin`.

    load_instance(instance, folder, log_method=None)
        Load instance from `folder`.
        * instance : object, instance to load, need to be initialized.
        * folder : str, folder to load from.
        * log_method : {None, function}, used as `log_method` parameter in
        `log_with_external_method` method of `LoggingMixin`.

    """

    delim = "^_^"
    dill_suffix = ".pkl"
    array_sub_folder = "__arrays"

    @staticmethod
    def _check_core(elem):
        if isinstance(elem, dict):
            if not Saving._check_dict(elem):
                return False
        if isinstance(elem, (list, tuple)):
            if not Saving._check_list_and_tuple(elem):
                return False
        if not Saving._check_elem(elem):
            return False
        return True

    @staticmethod
    def _check_elem(elem):
        if isinstance(elem, (type, np.generic, np.ndarray)):
            return False
        if callable(elem):
            return False
        try:
            json.dumps({"": elem})
            return True
        except TypeError:
            return False

    @staticmethod
    def _check_list_and_tuple(arr: Union[list, tuple]):
        for elem in arr:
            if not Saving._check_core(elem):
                return False
        return True

    @staticmethod
    def _check_dict(d: dict):
        for v in d.values():
            if not Saving._check_core(v):
                return False
        return True

    @staticmethod
    def save_dict(d: dict, name: str, folder: str) -> str:
        if Saving._check_dict(d):
            kwargs = {}
            suffix, method, mode = ".json", json.dump, "w"
        else:
            kwargs = {"recurse": True}
            suffix, method, mode = Saving.dill_suffix, dill.dump, "wb"
        file = os.path.join(folder, f"{name}{suffix}")
        with open(file, mode) as f:
            method(d, f, **kwargs)
        return os.path.abspath(file)

    @staticmethod
    def load_dict(name: str, folder: str = None):
        if folder is None:
            folder, name = os.path.split(name)
        name, suffix = os.path.splitext(name)
        if not suffix:
            json_file = os.path.join(folder, f"{name}.json")
            if os.path.isfile(json_file):
                with open(json_file, "r") as f:
                    return json.load(f)
            dill_file = os.path.join(folder, f"{name}{Saving.dill_suffix}")
            if os.path.isfile(dill_file):
                with open(dill_file, "rb") as f:
                    return dill.load(f)
        else:
            assert_msg = f"suffix should be either 'json' or 'pkl', {suffix} found"
            assert suffix in {".json", ".pkl"}, assert_msg
            name = f"{name}{suffix}"
            file = os.path.join(folder, name)
            if os.path.isfile(file):
                if suffix == ".json":
                    mode, load_method = "r", json.load
                else:
                    mode, load_method = "rb", dill.load
                with open(file, mode) as f:
                    return load_method(f)
        raise ValueError(f"config '{name}' is not found under '{folder}' folder")

    @staticmethod
    def deep_copy_dict(d: dict):
        tmp_folder = os.path.join(os.getcwd(), "___tmp_dict_cache___")
        if os.path.isdir(tmp_folder):
            shutil.rmtree(tmp_folder)
        os.makedirs(tmp_folder)
        dict_name = "deep_copy"
        Saving.save_dict(d, dict_name, tmp_folder)
        loaded_dict = Saving.load_dict(dict_name, tmp_folder)
        shutil.rmtree(tmp_folder)
        return loaded_dict

    @staticmethod
    def get_cache_file(instance):
        return f"{type(instance).__name__}.pkl"

    @staticmethod
    def save_instance(instance, folder, log_method=None):
        instance_str = str(instance)
        Saving.log_with_external_method(
            f"saving '{instance_str}' to '{folder}'",
            Saving.info_prefix,
            log_method,
            5,
        )

        def _record_array(k, v):
            extension = ".npy" if isinstance(v, np.ndarray) else ".lst"
            array_attribute_dict[f"{k}{extension}"] = v

        def _check_array(attr_key_, attr_value_, depth=0):
            if isinstance(attr_value_, dict):
                for k in list(attr_value_.keys()):
                    v = attr_value_[k]
                    extended_k = f"{attr_key_}{delim}{k}"
                    if isinstance(v, dict):
                        _check_array(extended_k, v, depth + 1)
                    elif isinstance(v, array_types):
                        _record_array(extended_k, v)
                        attr_value_.pop(k)
            if isinstance(attr_value_, array_types):
                _record_array(attr_key_, attr_value_)
                if depth == 0:
                    cache_excludes.add(attr_key_)

        main_file = Saving.get_cache_file(instance)
        instance_dict = shallow_copy_dict(instance.__dict__)
        verbose, cache_excludes = map(
            getattr,
            [instance] * 2,
            ["lock_verbose", "cache_excludes"],
            [False, set()],
        )
        if os.path.isdir(folder):
            if verbose:
                prefix = Saving.warning_prefix
                msg = f"'{folder}' will be cleaned up when saving '{instance_str}'"
                Saving.log_with_external_method(
                    msg, prefix, log_method, msg_level=logging.WARNING
                )
            shutil.rmtree(folder)
        save_path = os.path.join(folder, main_file)
        array_folder = os.path.join(folder, Saving.array_sub_folder)
        tuple(
            map(
                lambda folder_: os.makedirs(folder_, exist_ok=True),
                [folder, array_folder],
            )
        )
        sorted_attributes, array_attribute_dict = sorted(instance_dict), {}
        delim, array_types = Saving.delim, (list, np.ndarray)
        for attr_key in sorted_attributes:
            if attr_key in cache_excludes:
                continue
            attr_value = instance_dict[attr_key]
            _check_array(attr_key, attr_value)
        cache_excludes.add("_verbose_level_")
        with lock_manager(
            folder,
            [os.path.join(folder, main_file)],
            name=instance_str,
        ):
            with open(save_path, "wb") as f:
                d = {k: v for k, v in instance_dict.items() if k not in cache_excludes}
                dill.dump(d, f, recurse=True)
        if array_attribute_dict:
            sorted_array_files = sorted(array_attribute_dict)
            sorted_array_files_full_path = list(
                map(lambda f_: os.path.join(array_folder, f_), sorted_array_files)
            )
            with lock_manager(
                array_folder,
                sorted_array_files_full_path,
                name=f"{instance_str} (arrays)",
            ):
                for array_file, array_file_full_path in zip(
                    sorted_array_files, sorted_array_files_full_path
                ):
                    array_value = array_attribute_dict[array_file]
                    if array_file.endswith(".npy"):
                        np.save(array_file_full_path, array_value)
                    elif array_file.endswith(".lst"):
                        with open(array_file_full_path, "wb") as f:
                            np.save(f, array_value)
                    else:
                        raise ValueError(
                            f"unrecognized file type '{array_file}' occurred"
                        )

    @staticmethod
    def load_instance(instance, folder, *, log_method=None, verbose=True):
        if verbose:
            Saving.log_with_external_method(
                f"loading '{instance}' from '{folder}'",
                Saving.info_prefix,
                log_method,
                5,
            )
        with open(os.path.join(folder, Saving.get_cache_file(instance)), "rb") as f:
            instance.__dict__.update(dill.load(f))
        delim = Saving.delim
        array_folder = os.path.join(folder, Saving.array_sub_folder)
        for array_file in os.listdir(array_folder):
            attr_name, attr_ext = os.path.splitext(array_file)
            if attr_ext == ".npy":
                load_method = partial(np.load, allow_pickle=True)
            elif attr_ext == ".lst":

                def load_method(path):
                    return np.load(path, allow_pickle=True).tolist()

            else:
                raise ValueError(f"unrecognized file type '{array_file}' occurred")
            array_value = load_method(os.path.join(array_folder, array_file))
            attr_hierarchy = attr_name.split(delim)
            if len(attr_hierarchy) == 1:
                instance.__dict__[attr_name] = array_value
            else:
                hierarchy_dict = instance.__dict__
                for attr in attr_hierarchy[:-1]:
                    hierarchy_dict = hierarchy_dict.setdefault(attr, {})
                hierarchy_dict[attr_hierarchy[-1]] = array_value

    @staticmethod
    def prepare_folder(instance, folder):
        if os.path.isdir(folder):
            instance.log_msg(
                f"'{folder}' already exists, it will be cleared up to save our model",
                instance.warning_prefix,
                msg_level=logging.WARNING,
            )
            shutil.rmtree(folder)
        os.makedirs(folder)

    @staticmethod
    def compress(abs_folder, remove_original=True):
        shutil.make_archive(abs_folder, "zip", abs_folder)
        if remove_original:
            shutil.rmtree(abs_folder)

    @staticmethod
    def compress_loader(
        folder: str,
        is_compress: bool,
        *,
        remove_extracted: bool = True,
        logging_mixin: Optional[LoggingMixin] = None,
    ):
        class _manager(context_error_handler):
            def __enter__(self):
                if is_compress:
                    if os.path.isdir(folder):
                        msg = (
                            f"'{folder}' already exists, "
                            "it will be cleared up to load our model"
                        )
                        if logging_mixin is None:
                            print(msg)
                        else:
                            logging_mixin.log_msg(
                                msg,
                                logging_mixin.warning_prefix,
                                msg_level=logging.WARNING,
                            )
                        shutil.rmtree(folder)
                    with zipfile.ZipFile(f"{folder}.zip", "r") as zip_ref:
                        zip_ref.extractall(folder)

            def _normal_exit(self, exc_type, exc_val, exc_tb):
                if is_compress and remove_extracted:
                    shutil.rmtree(folder)

        return _manager()

