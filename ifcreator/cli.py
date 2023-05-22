import os
import click
# FastAPI是你的Web应用，而Uvicorn是运行你的Web应用的服务器
import uvicorn

from ifcreator import Focus
from ifcreator import opt_env_context
from iflearn.parameters import OPT


@click.group()
def main() -> None:
    pass


@main.command()
@click.option(
    "-p",
    "--port",
    default=8123,
    show_default=True,
    type=int,
    help="The port to listen on.",
)
@click.option(
    "--cpu",
    is_flag=True,
    default=False,
    show_default=True,
    type=bool,
    help="This flag will force the server to use cpu only.",
)
@click.option(
    "--focus",
    default="all",
    show_default=True,
    type=click.Choice([e.value for e in Focus]),
    help="""
Indicates which endpoints should we focus on, helpful if we only care about certain subset of features.
\n- all            |  will load all endpoints.
\n- sd             |  will load SD endpoints.
\n- sd.base        |  will only load basic SD endpoints, which means anime / inpainting / outpainting endpoints will not be loaded.
\n- sd.anime       |  will only load anime SD endpoints, which means basic / inpainting / outpainting endpoints will not be loaded.
\n- sd.inpainting  |  will only load inpainting / outpainting SD endpoints, which means basic / anime endpoints will not be loaded.
\n- sync           |  will only load 'sync' endpoints, which are relatively fast (e.g. sod, lama, captioning, harmonization, ...).
\n- control        |  will only load 'ControlNet' endpoints.
\n- pipeline       |  will only load 'Pipeline' endpoints.
\n-
""",
)
@click.option(
    "-r",
    "--reload",
    is_flag=True,
    default=False,
    show_default=True,
    type=bool,
    help="This flag represents the `reload` argument of `uvicorn.run`.",
)
@click.option(
    "-d",
    "--cache_dir",
    default="",
    show_default=True,
    type=str,
    help="Directory of the cache files.",
)
@click.option(
    "--lazy",
    is_flag=True,
    default=False,
    show_default=True,
    type=bool,
    help="This flag will enable the `lazy loading` mode.",
)
@click.option(
    "--limit",
    default=-1,
    show_default=True,
    type=int,
    help="Limitation of the pools (i.e. `WeightsPool` & `APIPool`).",
)
def serve(
    *,
    port: int,
    cpu: bool,
    focus: str,
    reload: bool,
    cache_dir: str,
    lazy: bool,
    limit: int,
) -> None:
    # 用来"增加"或"修改"一些默认配置的参数
    increment = dict(lazy_load=lazy, pool_limit=limit)
    if cpu:
        increment["cpu"] = True
    if focus != "all":
        increment["focus"] = focus
    iflearn_increment = {}
    if cache_dir:
        iflearn_increment["cache_dir"] = cache_dir
    # 两个上下文环境
    with opt_env_context(increment):
        # iflearn 的 OPT配置
        with OPT.opt_context(iflearn_increment):
            uvicorn.run(
                "ifcreator.apis.interface:app",
                host="0.0.0.0",
                port=port,
                reload=reload,
                reload_dirs=os.path.dirname(__file__) if reload else None,
            )
