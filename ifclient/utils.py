import json
import time
import requests
import logging

from typing import Dict, Any, Callable, Awaitable, TypeVar
from PIL import Image
from io import BytesIO
from pydantic import BaseModel
from iftool.web import raise_err
from aiohttp import ClientSession


async def get(url: str, session: ClientSession) -> bytes:
    async with session.get(url) as response:
        return await response.read()


def log_endpoint(endpoint: str, data: BaseModel) -> None:
    msg = f"{endpoint} endpoint entered with kwargs : {json.dumps(data.dict(), ensure_ascii=False)}"
    logging.debug(msg)


def log_times(endpoint: str, times: Dict[str, float]) -> None:
    times["__total__"] = sum(times.values())
    logging.debug(f"elapsed time of endpoint {endpoint} : {json.dumps(times)}")


async def run_algorithm(algorithm: Any, data: BaseModel, *args: Any) -> BaseModel:
    try:
        return await algorithm.run(data, *args)
    except Exception as err:
        raise_err(err)


async def _download(session: ClientSession, url: str) -> bytes:
    try:
        return await get(url, session)
    except Exception:
        return requests.get(url).content


async def _download_image(session: ClientSession, url: str) -> Image.Image:
    raw_data = None
    try:
        raw_data = await _download(session, url)
        return Image.open(BytesIO(raw_data))
    except Exception as err:
        if raw_data is None:
            msg = f"raw | None | err | {err}"
        else:
            try:
                msg = raw_data.decode("utf-8")
            except:
                msg = f"raw | {raw_data[:20]} | err | {err}"
        raise ValueError(msg)

TRes = TypeVar("TRes")


async def _download_with_retry(
    download_fn: Callable[[ClientSession, str], Awaitable[TRes]],
    session: ClientSession,
    url: str,
    retry: int = 3,
    interval: int = 1,
) -> TRes:
    msg = ""
    for i in range(retry):
        try:
            res = await download_fn(session, url)
            if i > 0:
                logging.warning(f"succeeded after {i} retries")
            return res
        except Exception as err:
            msg = str(err)
        time.sleep(interval)
    raise ValueError(f"{msg}\n(After {retry} retries)")


async def download_with_retry(
    session: ClientSession,
    url: str,
    *,
    retry: int = 3,
    interval: int = 1,
) -> bytes:
    return await _download_with_retry(_download, session, url, retry, interval)


async def download_image_with_retry(
    session: ClientSession,
    url: str,
    *,
    retry: int = 3,
    interval: int = 1,
) -> Image.Image:
    return await _download_with_retry(_download_image, session, url, retry, interval)


__all__ = [
    "get",
    "log_times",
    "log_endpoint",
    "run_algorithm",
    "download_with_retry",
    "download_image_with_retry",
]