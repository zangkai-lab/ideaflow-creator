from aiohttp import ClientSession
"""
PIL是Python Imaging Library的简称，是一个创建、打开、操作图片的Python库
它支持大量的图片格式，提供了强大的图像处理功能
"""
from PIL import Image

from .parameters import use_cos
from ifclient.utils import download_with_retry as download
from ifclient.utils import download_image_with_retry as download_image

RETRY = 3
CDN_HOST = "https://ailabcdn.nolibox.com"
COS_HOST = "https://ailab-1310750649.cos.ap-shanghai.myqcloud.com"


"""
COS代表云对象存储(Cloud Object Storage)，是一种存储服务，它能让用户通过网络远程存储和读取数据
CDN代表内容分发网络(Content Delivery Network)，是一种通过在各地放置节点缓存内容，使用户能够就近获取数据，从而加速数据获取速度的技术
"""


async def download_with_retry(
    session: ClientSession,
    url: str,
    *,
    retry: int = RETRY,
    interval: int = 1,
) -> bytes:
    if use_cos() and url.startswith(CDN_HOST):
        url = url.replace(CDN_HOST, COS_HOST)
    return await download(session, url, retry=retry, interval=interval)


async def download_image_with_retry(
    session: ClientSession,
    url: str,
    *,
    retry: int = RETRY,
    interval: int = 1,
) -> Image.Image:
    if use_cos() and url.startswith(CDN_HOST):
        url = url.replace(CDN_HOST, COS_HOST)
    return await download_image(session, url, retry=retry, interval=interval)


__all__ = [
    "download_with_retry",
    "download_image_with_retry",
]