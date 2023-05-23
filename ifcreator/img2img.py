from pydantic import BaseModel, Field
from typing import Optional, Tuple

from .common import Img2ImgDiffusionModel, ReturnArraysModel, HighresModel


img2img_sd_endpoint = "/img2img/sd"


class _Img2ImgSDModel(BaseModel):
    text: str = Field(..., description="The text that we want to handle.")
    fidelity: float = Field(
        0.2,
        ge=0.0,
        le=1.0,
        description="The fidelity of the input image.",
    )
    keep_alpha: bool = Field(
        True,
        description="""
Whether the returned image should keep the alpha-channel of the input image or not.
> If the input image is a sketch image, then `keep_alpha` needs to be False in most of the time.  
""",
    )
    wh: Tuple[int, int] = Field(
        (0, 0),
        description="The output size, `0` means as-is",
    )
    highres_info: Optional[HighresModel] = Field(None, description="Highres info.")


class Img2ImgSDModel(ReturnArraysModel, Img2ImgDiffusionModel, _Img2ImgSDModel):
    pass


__all__ = [
    "img2img_sd_endpoint",
    "Img2ImgSDModel",
]