"""
    和cli配套的参数配置文件
"""
import os
import json

from enum import Enum
from typing import Any, Dict
from fastapi import Response
from iftool.misc import shallow_copy_dict

OPT = dict(
    # 控制输出日志的详细程度，True是详细模式，False是简单模式
    verbose=True,
    cpu=False,
    # cos是什么东西？
    use_cos=True,
    request_domain="localhost",
    # 这两个选项可能用于设置Redis服务器的参数。redis_kwarg是本地使用的的Redis服务器，audit_redis_kwargs可能是设置审计用的Redis远程服务器
    redis_kwargs=dict(host="localhost", port=6379, db=0),
    audit_redis_kwargs=dict(host="172.17.16.7", port=6379, db=1),
    bypass_audit=False,
    # 这个选项用于设置Kafka服务器的参数
    kafka_server="0.0.0.0:80",
    kafka_topic="creator",
    kafka_group_id="creator-consumer-1",
    kafka_max_poll_records=1,
    kafka_max_poll_interval_ms=5 * 60 * 1000,
    pending_queue_key="KAFKA_PENDING_QUEUE",
)

OPT_ENV_KEY = "IFCREATOR_ENV"

"""
opt_context类用于临时修改全局的OPT字典。
在__enter__方法中，它将OPT字典更新为increment字典中的键值对。
在__exit__方法中，它将OPT字典恢复为原来的状态。
这意味着，你可以使用opt_context来临时修改OPT字典，而不用担心这些修改会影响到其他的代码
"""


class opt_context:
    def __init__(self, increment: Dict[str, Any]) -> None:
        self._increment = increment
        self._backup = shallow_copy_dict(OPT)

    def __enter__(self) -> None:
        OPT.update(self._increment)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        OPT.update(self._backup)


class opt_env_context:
    def __init__(self, increment: Dict[str, Any]) -> None:
        self._increment = increment
        self._backup = os.environ.get(OPT_ENV_KEY)

    def __enter__(self) -> None:
        os.environ[OPT_ENV_KEY] = json.dumps(self._increment)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._backup is None:
            del os.environ[OPT_ENV_KEY]
        else:
            os.environ[OPT_ENV_KEY] = self._backup


# 设置一个枚举类
class Focus(str, Enum):
    ALL = "all"
    SD = "sd"
    SD_BASE = "sd.base"
    SD_ANIME = "sd.anime"
    SD_INPAINTING = "sd.inpainting"
    SYNC = "sync"
    CONTROL = "control"
    PIPELINE = "pipeline"


def verbose() -> bool:
    return OPT["verbose"]


def get_focus() -> Focus:
    return OPT.get("focus", "all")


def lazy_load() -> bool:
    return OPT.get("lazy_load", False)


def pool_limit() -> int:
    return OPT.get("pool_limit", -1)


def use_cos() -> bool:
    return OPT["use_cos"]


def inject_headers(response: Response) -> None:
    response.headers["X-Request-Domain"] = OPT["request_domain"]


def redis_kwargs() -> Dict[str, Any]:
    return shallow_copy_dict(OPT["redis_kwargs"])


def audit_redis_kwargs() -> Dict[str, Any]:
    return shallow_copy_dict(OPT["audit_redis_kwargs"])


def bypass_audit() -> bool:
    return OPT["bypass_audit"]


def kafka_server() -> str:
    return OPT["kafka_server"]


def kafka_topic() -> str:
    return OPT["kafka_topic"]


def kafka_group_id() -> str:
    return OPT["kafka_group_id"]


def kafka_max_poll_records() -> int:
    return OPT["kafka_max_poll_records"]


def kafka_max_poll_interval_ms() -> int:
    return OPT["kafka_max_poll_interval_ms"]


def get_pending_queue_key() -> str:
    return OPT["pending_queue_key"]


"""
    当其他模块使用from module import *这样的语法导入一个模块时，只有__all__中列出的对象会被导入。
    如果一个模块没有定义__all__，那么from module import *会导入该模块中所有不以下划线开头的全局名称
"""

__all__ = [
    "OPT",
    "OPT_ENV_KEY",
    "opt_context",
    "opt_env_context",
    "Focus",
    "use_cos",
    "verbose",
    "get_focus",
    "inject_headers",
    "redis_kwargs",
    "audit_redis_kwargs",
    "kafka_server",
    "kafka_topic",
    "kafka_group_id",
    "kafka_max_poll_records",
    "kafka_max_poll_interval_ms",
    "get_pending_queue_key",
]
