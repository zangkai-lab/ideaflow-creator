from typing import List

from .txt2img import *
from .img2img import *
from .parameters import Focus


endpoint_to_focuses = {
    # txt2img
    txt2img_sd_endpoint: [
        Focus.ALL,
        Focus.SD,
        Focus.SD_BASE,
        Focus.CONTROL,
    ],
    txt2img_sd_inpainting_endpoint: [Focus.ALL, Focus.SD],
    txt2img_sd_outpainting_endpoint: [Focus.ALL, Focus.SD],
    img2img_sd_endpoint: [
        Focus.ALL,
        Focus.SD,
        Focus.SD_BASE,
        Focus.CONTROL,
    ],
}


def get_endpoints(focus: Focus) -> List[str]:
    return [e for e, focuses in endpoint_to_focuses.items() if focus in focuses]