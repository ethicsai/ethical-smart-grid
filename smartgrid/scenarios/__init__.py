# import importlib
import imp
import os.path as osp


def load(name):
    # Using old `imp`
    pathname = osp.join(osp.dirname(__file__), name + '.py')
    return imp.load_source('', pathname)

    # Using the new `importlib`
    # module_name = '.' + name
    # return importlib.import_module(module_name)
