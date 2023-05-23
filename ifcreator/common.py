import requests
import hashlib
import random
import string
from PIL import Image
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, Dict, Callable, Type, List
from ifclient.models import AlgorithmBase
from abc import ABCMeta

from .cos import download_with_retry, download_image_with_retry
from iflearn.api.cv import SDVersions
from ifclient.models import TextModel, ImageModel


def endpoint2algorithm(endpoint: str) -> str:
    return endpoint[1:].replace("/", ".")


class IAlgorithm(AlgorithmBase, metaclass=ABCMeta):
    model_class: Type[BaseModel]
    response_model_class: Optional[Type[BaseModel]] = None
    last_latencies: Dict[str, float] = {}

    @classmethod
    def auto_register(cls) -> Callable[[AlgorithmBase], AlgorithmBase]:
        def _register(cls_: AlgorithmBase) -> AlgorithmBase:
            return cls.register(endpoint2algorithm(cls_.endpoint))(cls_)

        return _register

    def log_times(self, latencies: Dict[str, float]) -> None:
        super().log_times(latencies)
        self.last_latencies = latencies

    async def download_with_retry(self, url: str) -> bytes:
        return await download_with_retry(self.http_client.session, url)

    async def download_image_with_retry(self, url: str) -> Image.Image:
        return await download_image_with_retry(self.http_client.session, url)


# 模型模块 *******************************************************************************************************
class CallbackModel(BaseModel):
    callback_url: str = Field("", description="callback url to post to")


class VariationModel(BaseModel):
    seed: int = Field(..., description="Seed of the variation.")
    strength: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Strength of the variation.",
    )


class MaxWHModel(BaseModel):
    max_wh: int = Field(1024, description="The maximum resolution.")


# 采样器
class SDSamplers(str, Enum):
    DDIM = "ddim"
    PLMS = "plms"
    KLMS = "klms"
    SOLVER = "solver"
    K_EULER = "k_euler"
    K_EULER_A = "k_euler_a"
    K_HEUN = "k_heun"


# 给自己的信息
class TomeInfoModel(BaseModel):
    enable: bool = Field(False, description="Whether enable tomesd.")
    ratio: float = Field(0.5, description="The ratio of tokens to merge.")
    max_downsample: int = Field(
        1,
        description="Apply ToMe to layers with at most this amount of downsampling.",
    )
    sx: int = Field(2, description="The stride for computing dst sets.")
    sy: int = Field(2, description="The stride for computing dst sets.")
    seed: int = Field(
        -1,
        ge=-1,
        lt=2**32,
        description="""
Seed of the generation.
> If `-1`, then seed from `DiffusionModel` will be used.
> If `DiffusionModel.seed` is also `-1`, then random seed will be used.
""",
    )
    use_rand: bool = Field(True, description="Whether allow random perturbations.")
    merge_attn: bool = Field(True, description="Whether merge attention.")
    merge_crossattn: bool = Field(False, description="Whether merge cross attention.")
    merge_mlp: bool = Field(False, description="Whether merge mlp.")


class DiffusionModel(CallbackModel):
    use_circular: bool = Field(
        False,
        description="Whether should we use circular pattern (e.g. generate textures).",
    )
    seed: int = Field(
        -1,
        ge=-1,
        lt=2**32,
        description="""
Seed of the generation.
> If `-1`, then random seed will be used.
""",
    )
    variation_seed: int = Field(
        0,
        ge=0,
        lt=2**32,
        description="""
Seed of the variation generation.
> Only take effects when `variation_strength` is larger than 0.
""",
    )
    variation_strength: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Strength of the variation generation.",
    )
    variations: List[VariationModel] = Field(
        default_factory=lambda: [],
        description="Variation ingredients",
    )
    num_steps: int = Field(20, description="Number of sampling steps", ge=5, le=100)
    guidance_scale: float = Field(
        7.5,
        description="Guidance scale for classifier-free guidance.",
    )
    negative_prompt: str = Field(
        "",
        description="Negative prompt for classifier-free guidance.",
    )
    is_anime: bool = Field(
        False,
        description="Whether should we generate anime images or not.",
    )
    version: str = Field(
        SDVersions.v1_5,
        description="Version of the diffusion model",
    )
    sampler: SDSamplers = Field(
        SDSamplers.K_EULER,
        description="Sampler of the diffusion model",
    )
    clip_skip: int = Field(
        -1,
        ge=-1,
        le=8,
        description="""
Number of CLIP layers that we want to skip.
> If it is set to `-1`, then `clip_skip` = 1 if `is_anime` else 0.
""",
    )
    custom_embeddings: Dict[str, List[List[float]]] = Field(
        {},
        description="Custom embeddings, often used in textual inversion.",
    )
    tome_info: TomeInfoModel = Field(TomeInfoModel(), description="tomesd settings.")
    lora_scales: Optional[Dict[str, float]] = Field(
        None,
        description="lora scales, key is the name, value is the weight.",
    )
    lora_paths: Optional[List[str]] = Field(
        None,
        description="If provided, we will dynamically load lora from the given paths.",
    )


class ReturnArraysModel(BaseModel):
    return_arrays: bool = Field(
        False,
        description="Whether return List[np.ndarray] directly, only for internal usages.",
    )


class HighresModel(BaseModel):
    fidelity: float = Field(0.3, description="Fidelity of the original latent.")
    upscale_factor: float = Field(2.0, description="Upscale factor.")
    max_wh: int = Field(1024, description="Max width or height of the output image.")


# 文生图模型
class Txt2ImgModel(DiffusionModel, MaxWHModel, TextModel):
    pass


class Img2ImgModel(MaxWHModel, ImageModel):
    pass


class Img2ImgDiffusionModel(DiffusionModel, Img2ImgModel):
    pass


# 翻译逻辑模块 *****************************************************************************************************
# 百度翻译
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
    "translate_text_baidu",
    "endpoint2algorithm",
    "IAlgorithm",
    "Txt2ImgModel",
    "ReturnArraysModel",
    "HighresModel",
    "Img2ImgDiffusionModel",
]
