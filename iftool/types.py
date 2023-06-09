import numpy as np
from typing import Dict, Union, Any, Optional, List, NamedTuple, Callable

try:
    import torch
    import torchvision
    import torch.nn.functional as F
except:

    class _torch(NamedTuple):
        device: Any
        Tensor: Any
        from_numpy: Callable

    class _torchvision_utils(NamedTuple):
        make_grid: Callable
        save_image: Callable

    class _torchvision(NamedTuple):
        utils: _torchvision_utils

    torch = _torch(None, None, lambda: None)
    torchvision = _torchvision_utils(lambda: None, lambda: None)
    F = None


general_config_type = Optional[Union[str, Dict[str, Any]]]
configs_type = Optional[Union[List[Dict[str, Any]], Dict[str, Any]]]

arr_type = Union[np.ndarray, torch.Tensor]
np_dict_type = Dict[str, Union[np.ndarray, Any]]
tensor_dict_type = Dict[str, Union[torch.Tensor, Any]]