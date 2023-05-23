import torch

from enum import Enum
from typing import Optional, Any, Dict, Protocol

from .parameters import pool_limit, OPT, lazy_load
from iftool.data_structures import ILoadableItem, ILoadablePool


class APIs(str, Enum):
    SD = "sd"
    SD_INPAINTING = "sd_inpainting"
    ESR = "esr"
    ESR_ANIME = "esr_anime"
    INPAINTING = "inpainting"
    LAMA = "lama"
    SEMANTIC = "semantic"
    HRNET = "hrnet"
    ISNET = "isnet"
    BLIP = "blip"
    PROMPT_ENHANCE = "prompt_enhance"


class IAPI:
    def to(self, device: str, *, use_half: bool) -> None:
        pass


class APIInit(Protocol):
    def __call__(self, init_to_cpu: bool) -> IAPI:
        pass


class LoadableAPI(ILoadableItem[IAPI]):
    def __init__(
        self,
        init_fn: APIInit,
        *,
        init: bool = False,
        force_not_lazy: bool = False,
        has_annotator: bool = False,
    ):
        super().__init__(lambda: init_fn(self.init_to_cpu), init=init)
        self.force_not_lazy = force_not_lazy
        self.has_annotator = has_annotator

    @property
    def lazy(self) -> bool:
        return lazy_load() and not self.force_not_lazy

    @property
    def init_to_cpu(self) -> bool:
        return self.lazy or OPT["cpu"]

    @property
    def need_change_device(self) -> bool:
        return self.lazy and not OPT["cpu"]

    @property
    def annotator_kwargs(self) -> Dict[str, Any]:
        return {"no_annotator": True} if self.has_annotator else {}

    def load(self, *, no_change: bool = False, **kwargs: Any) -> IAPI:
        super().load()
        if not no_change and self.need_change_device:
            self._item.to("cuda:0", use_half=True, **self.annotator_kwargs)
        return self._item

    def cleanup(self) -> None:
        if self.need_change_device:
            self._item.to("cpu", use_half=False, **self.annotator_kwargs)
            torch.cuda.empty_cache()

    def unload(self) -> None:
        self.cleanup()
        return super().unload()


class APIPool(ILoadablePool[IAPI]):
    def register(self, key: str, init_fn: APIInit) -> None:
        def _init(init: bool) -> LoadableAPI:
            kw = dict(
                force_not_lazy=key in (APIs.SD, APIs.SD_INPAINTING),
                has_annotator=key in (APIs.SD, APIs.SD_INPAINTING),
            )
            api = LoadableAPI(init_fn, init=False, **kw)
            if init:
                print("> init", key, "(lazy)" if api.lazy else "")
                api.load(no_change=api.lazy)
            return api

        if key in self:
            return
        return super().register(key, _init)

    def cleanup(self, key: str) -> None:
        loadable_api: Optional[LoadableAPI] = self.pool.get(key)
        if loadable_api is None:
            raise ValueError(f"key '{key}' does not exist")
        loadable_api.cleanup()

    def need_change_device(self, key: str) -> bool:
        loadable_api: Optional[LoadableAPI] = self.pool.get(key)
        if loadable_api is None:
            raise ValueError(f"key '{key}' does not exist")
        return loadable_api.need_change_device

    def update_limit(self) -> None:
        self.limit = pool_limit()


api_pool = APIPool()