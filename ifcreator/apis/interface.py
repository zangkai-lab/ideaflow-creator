import os
from fastapi import FastAPI
from enum import Enum
from typing import Dict
from translate import Translator

from ifclient.models import *
from ifclient.utils import run_algorithm
from ifclient.core import HttpClient
from ifclient.core import TritonClient
from iftool.web import get_responses

from ifcreator import *

"""
Pydantic 是一个 Python 库，用于数据解析和验证。它使用 Python 类型注解来验证、序列化和反序列化复杂的数据类型，如字典、JSON等。
"""
from pydantic import BaseModel

app = FastAPI()
root = os.path.dirname(__file__)

constants = dict(
    triton_host=None,
    triton_port=8000,
    model_root=os.path.join(root, "models"),
    token_root=os.path.join(root, "tokens"),
)


# clients
# http client
http_client = HttpClient()
# triton client
"""
Triton Inference Server（前称为TensorRT Inference Server）是NVIDIA提供的开源软件，
它可以部署在数据中心的AI模型，以提高利用率并实现更好的性能。
Triton Inference Server支持AI模型的各种框架，包括TensorFlow、TensorRT、PyTorch等。
"""
triton_host = constants["triton_host"]
if triton_host is None:
    triton_client = None
else:
    triton_client = TritonClient(url=f"{triton_host}:{constants['triton_port']}")
# collect
clients = dict(
    http=http_client,
    triton=triton_client,
)

# algorithms
all_algorithms: Dict[str, AlgorithmBase] = {
    k: v(clients) for k, v in algorithms.items()
}


# 健康检查
class HealthStatus(Enum):
    ALIVE = "alive"


class HealthCheckResponse(BaseModel):
    status: HealthStatus


@app.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    return {"status": "alive"}


# demo:一个通用模型加载器
@app.post(demo_hello_endpoint, responses=get_responses(HelloResponse))
async def hello(data: HelloModel) -> HelloResponse:
    return await run_algorithm(all_algorithms["demo.hello"], data)


# get prompt
def translate_text(text: str) -> str:
    translator = Translator(to_lang="en")
    translation = translator.translate(text)
    return translation


@app.post("/translate", responses=get_responses(GetPromptResponse))
@app.post("/get_prompt", responses=get_responses(GetPromptResponse))
def get_prompt(data: GetPromptModel) -> GetPromptResponse:
    if data.need_translate:
        try:
            translated_text = translate_text(data.text)
            return GetPromptResponse(text=translated_text, success=True, reason="")
        except Exception as e:
            return GetPromptResponse(text="", success=False, reason=str(e))
    else:
        return GetPromptResponse(text=data.text, success=True, reason="")


"""
这部分的代码只有在直接运行这个Python文件时才会执行。如果这个文件被其他Python文件导入，那么这部分的代码就不会执行
"""
# if __name__ == "__main__":
#     import uvicorn
#
#     # --reload参数使得服务器在你每次保存文件后自动重启
#     uvicorn.run("interface:app", host="0.0.0.0", port=8000, reload=True)
