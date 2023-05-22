import requests
import hashlib
import random
import string
from pydantic import BaseModel, Field


def translate_text_baidu(text: str, appid: str, secret_key: str) -> str:
    api_url = "https://fanyi-api.baidu.com/api/trans/vip/translate"
    from_lang = "zh"
    to_lang = "en"
    salt = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

    sign = appid + text + salt + secret_key
    md5 = hashlib.md5()
    md5.update(sign.encode('utf-8'))
    sign = md5.hexdigest()

    params = {
        "q": text,
        "from": from_lang,
        "to": to_lang,
        "appid": appid,
        "salt": salt,
        "sign": sign
    }

    response = requests.get(api_url, params=params)
    response_json = response.json()

    if "trans_result" in response_json:
        return response_json["trans_result"][0]["dst"]
    else:
        return "Error: " + response_json["error_msg"]


class GetPromptModel(BaseModel):
    text: str
    need_translate: bool = Field(
        True,
        description="Whether we need to translate the input text.",
    )


class GetPromptResponse(BaseModel):
    text: str
    success: bool
    reason: str


__all__ = [
    "GetPromptResponse",
    "GetPromptModel",
    "translate_text_baidu"
]
