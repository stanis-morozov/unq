"""
Does all sorts of dark magic in order to build/import c++ bfs
"""
import os
import os.path as osp
import setuptools.sandbox

package_abspath = osp.join(*osp.split(osp.abspath(__file__))[:-1])
try:
    from . import index_quantization_ranker
except Exception as e:
    # try build
    workdir = os.getcwd()
    try:
        os.chdir(package_abspath)
        setuptools.sandbox.run_setup(osp.join(package_abspath, 'setup.py'), ['clean', 'build'])
        os.system('cp {}/build/lib*/*.so {}/.'.format(package_abspath, package_abspath))
    except Exception as e:
        raise ImportError("Failed to import index_quantization_ranker, please see error log or compile manually")
    finally:
        os.chdir(workdir)

from . import index_quantization_ranker
IndexQuantizationRankerBase = index_quantization_ranker.IndexQuantizationRanker