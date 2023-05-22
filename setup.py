"""
    项目的安装配置文件
    该项目基于FastAPI的客户端框架，使用PyTorch作为AI引擎，支持与Triton推理服务器进行交互
"""
from setuptools import setup, find_packages

PACKAGE_NAME = "ideaflow-creator"
VERSION = "0.0.1"
DESCRIPTION = "An AI-powered creator for everyone."
with open("README.md", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,

    author="zangkai-lab",
    author_email="erickant505@gmail.com",
    long_description_content_type="text/markdown",
    keywords="python pytorch",

    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    # 暴露为命令行工具
    entry_points={"console_scripts": ["ifcreator = ifcreator.cli:main"]},
    install_requires=[
        "numpy",
        # 提供了一种简单的方式来定义命令，处理命令行参数，以及显示帮助信息
        "click",
        # FastAPI的性能更高。根据官方的基准测试，FastAPI的性能接近于NodeJS和Go，远高于传统的Python框架，如Flask和Django。
        "fastapi",
        # Uvicorn是运行Web应用的服务器
        "uvicorn",
        # Pydantic 是一个 Python 库，用于数据解析和验证。它使用 Python 类型注解来验证、序列化和反序列化复杂的数据类型，如字典、JSON等。
        "pydantic",
        # 是一个高性能的并发 HTTP 客户端库，它使用 gevent 这个 Python 库来实现并发
        "geventhttpclient",
        # 用于异步 HTTP 网络请求的 Python 库
        "aiohttp",
    ],
    # 额外依赖项通常是在某些特定的环境或者使用场景下才需要的，这里是在多用户的生产环境才需要
    extras_require={
        "kafka": [
            # 用户数据流处理平台
            "kafka-python",
            "redis[hiredis]",
            # 腾讯云对象存储服务（COS）的Python SDK
            "cos-python-sdk-v5",
        ]
    },

)
