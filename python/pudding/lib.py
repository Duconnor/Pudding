'''
This file is responsible for loading the C shared library
'''
import ctypes

def _find_lib_path() -> str:
    return ""

def _load_lib() -> ctypes.CDLL:
    lib_path = _find_lib_path()
    lib = ctypes.CDLL(lib_path)
    return lib

_LIB = _load_lib()
