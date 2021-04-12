"""
@file
@brief  Main file.
"""
import os

def _find_():
    _this_ = os.path.abspath(os.path.dirname(__file__))
    for k in os.listdir(_this_):
        ext = os.path.splitext(k)[-1]
        if ext in {'.dll', '.so'}:
            return os.path.join(_this_, k)
    raise FileNotFoundError(
        "Unable to find any library in %r." % _this_)
    
    
_name_ = _find_()


def get_library_path():
    """
    The custom operator library binary path
    :return: A string of the this library path.
    """
    return _name_
    
