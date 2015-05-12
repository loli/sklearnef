import os

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    config = Configuration('sklearnef', parent_package, top_path)

    config.add_subpackage("ensemble")
    config.add_subpackage("ensemble/tests")
    config.add_subpackage("tree")
    config.add_subpackage("tree/tests")

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
