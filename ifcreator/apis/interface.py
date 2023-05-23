import os
import torch
import json
from fastapi import FastAPI, Response
from enum import Enum
from typing import Dict

from ifclient.models import *
from ifclient.utils import run_algorithm
from ifclient.core import HttpClient
from ifclient.core import TritonClient
from iftool.web import get_responses, get_image_response_kwargs
from iftool.types import tensor_dict_type

from ifcreator import *
from ifcreator.utils import api_pool

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
@app.post("/translate", responses=get_responses(GetPromptResponse))
@app.post("/get_prompt", responses=get_responses(GetPromptResponse))
def get_prompt(data: GetPromptModel) -> GetPromptResponse:
    if data.need_translate:
        try:
            translated_text = translate_text_baidu(data.text, "", "")
            return GetPromptResponse(text=translated_text, success=True, reason="")
        except Exception as e:
            return GetPromptResponse(text="", success=False, reason=str(e))
    else:
        return GetPromptResponse(text=data.text, success=True, reason="")


# switch local checkpoint
class ModelRootResponse(BaseModel):
    root: str


@app.post("/model_root", responses=get_responses(GetPromptResponse))
def get_model_root() -> ModelRootResponse:
    return ModelRootResponse(root=constants["model_root"])


# inject custom tokens: 注入自定义令牌
custom_embeddings: tensor_dict_type = {}


def _inject_custom_tokens(root: str) -> tensor_dict_type:
    local_customs: tensor_dict_type = {}
    if not os.path.isdir(root):
        return local_customs
    for file in os.listdir(root):
        try:
            path = os.path.join(root, file)
            d = torch.load(path, map_location="cpu")
            local_customs.update({k: v.tolist() for k, v in d.items()})
        except:
            continue
    if local_customs:
        print(f"> Following tokens are loaded: {', '.join(sorted(local_customs))}")
        custom_embeddings.update(local_customs)
    return local_customs


# meta
env_opt_json = os.environ.get(OPT_ENV_KEY)
if env_opt_json is not None:
    OPT.update(json.loads(env_opt_json))
focus = get_focus()
registered_algorithms = set()
api_pool.update_limit()


def register_endpoint(endpoint: str) -> None:
    name = endpoint[1:].replace("/", "_")
    algorithm_name = endpoint2algorithm(endpoint)
    algorithm: IAlgorithm = all_algorithms[algorithm_name]
    registered_algorithms.add(algorithm_name)
    data_model = algorithm.model_class
    if algorithm.response_model_class is None:
        response_model = Response
        post_kwargs = get_image_response_kwargs()
    else:
        response_model = algorithm.response_model_class
        post_kwargs = dict(responses=get_responses(response_model))

    @app.post(endpoint, **post_kwargs, name=name)
    async def _(data: data_model) -> response_model:
        if (
                isinstance(data, (Txt2ImgSDModel, Img2ImgSDModel))
                and not data.custom_embeddings
        ):
            data.custom_embeddings = {
                token: embedding
                for token, embedding in custom_embeddings.items()
                if token in data.text or token in data.negative_prompt
            }
        return await run_algorithm(algorithm, data)


for endpoint in get_endpoints(focus):
    register_endpoint(endpoint)


@app.on_event("startup")
async def startup() -> None:
    http_client.start()
    OPT["use_cos"] = False
    for k, v in all_algorithms.items():
        if k in registered_algorithms:
            v.initialize()
    _inject_custom_tokens(constants["token_root"])
    print("> Server is Ready!")


@app.on_event("shutdown")
async def shutdown() -> None:
    await http_client.stop()


"""
这部分的代码只有在直接运行这个Python文件时才会执行。如果这个文件被其他Python文件导入，那么这部分的代码就不会执行
"""
# if __name__ == "__main__":
#     import uvicorn
#
#     # --reload参数使得服务器在你每次保存文件后自动重启
#     uvicorn.run("interface:app", host="0.0.0.0", port=8000, reload=True)
