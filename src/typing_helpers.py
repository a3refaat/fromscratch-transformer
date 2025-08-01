# typing_helpers.py
from typing import TYPE_CHECKING, Union
import numpy as np

if TYPE_CHECKING:
    import cupy

ArrayType = Union[np.ndarray, "cupy.ndarray"]