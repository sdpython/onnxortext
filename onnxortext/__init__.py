# coding: utf-8
"""
@file
@brief The entry point to onnxruntime custom op library.
"""

__version__ = "0.0.1"
__author__ = "Xavier Dupré"


from .helper import (
    get_library_path, Opdef, PyCustomOpDef)

onnx_op = Opdef.declare
PyOp = PyCustomOpDef
