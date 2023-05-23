from pydantic import BaseModel, Field
from typing import Optional

from .common import Txt2ImgModel, ReturnArraysModel, HighresModel


txt2img_sd_endpoint = "/txt2img/sd"
txt2img_sd_inpainting_endpoint = "/txt2img/sd.inpainting"
txt2img_sd_outpainting_endpoint = "/txt2img/sd.outpainting"


class _Txt2ImgSDModel(BaseModel):
    w: int = Field(512, description="The desired output width.")
    h: int = Field(512, description="The desired output height.")
    highres_info: Optional[HighresModel] = Field(None, description="Highres info.")


class Txt2ImgSDModel(ReturnArraysModel, Txt2ImgModel, _Txt2ImgSDModel):
    pass


__all__ = [
    "Txt2ImgSDModel",
    "txt2img_sd_endpoint",
    "txt2img_sd_inpainting_endpoint",
    "txt2img_sd_outpainting_endpoint",
]