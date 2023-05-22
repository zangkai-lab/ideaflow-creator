from typing import Any
from typing import Dict
from typing import Type
from typing import Optional

from .constants import WEB_ERR_CODE

try:
    from fastapi import Response
    from fastapi import HTTPException
    from pydantic import BaseModel
except:
    Response = HTTPException = None
    BaseModel = object


class RuntimeError(BaseModel):
    detail: str

    class Config:
        schema_extra = {
            "example": {"detail": "RuntimeError occurred."},
        }


def get_responses(
        success_model: Type[BaseModel],
        *,
        json_example: Optional[Dict[str, Any]] = None,
) -> Dict[int, Dict[str, Type]]:
    success_response: Dict[str, Any] = {"model": success_model}
    if json_example is not None:
        content = success_response["content"] = {}
        json_field = content["application/json"] = {}
        json_field["example"] = json_example
    return {
        200: success_response,
        WEB_ERR_CODE: {"model": RuntimeError},
    }
