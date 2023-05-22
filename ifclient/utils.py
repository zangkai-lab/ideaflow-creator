import json
import logging

from typing import Dict, Any
from pydantic import BaseModel


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