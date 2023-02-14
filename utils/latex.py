import torch
import numpy as np
from typing import Union


def numpy_to_latex_bmatrix(a: Union[np.ndarray, torch.Tensor], decimals: int = 2) -> None:
    """Prints a LaTeX bmatrix

    :a: numpy array
    :decimals: for rounding
    :returns: None
    """
    if isinstance(a, torch.Tensor):
        a = a.detach().numpy()
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    a = a.round(decimals)
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv += [r'\end{bmatrix}']
    print('\n'.join(rv))
