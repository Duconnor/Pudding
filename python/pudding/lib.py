'''
This file is responsible for loading the C shared library
'''
import ctypes
import os

def _find_lib_path() -> str:
    '''
    Assume the compiled shared library lies in build/src under the root directory

    For now, only support Linux
    '''
    this_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    lib_path = os.path.join(this_path, 'build', 'src', 'libpudding.so')
    if os.path.exists(lib_path) and os.path.isfile(lib_path):
        return lib_path
    else:
        raise RuntimeError('Cannot find the shared library (libpudding.so)')


def _load_lib() -> ctypes.CDLL:
    lib_path = _find_lib_path()
    lib = ctypes.CDLL(lib_path)
    return lib

_LIB = _load_lib()
CONTIGUOUS_FLAG = 'C_CONTIGUOUS'
