from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
      ext_modules=cythonize('./rcit/motion/motion_interpolation/motion_interp_general.pyx'),
      include_dirs=[numpy.get_include()],)

setup(
      ext_modules=cythonize('./rcit/motion/motion_interpolation/motion_interp_dace.pyx'),
      extra_compile_args=["-fopenmp"],
      extra_link_args=["-fopenmp"],
      include_dirs=[numpy.get_include()],)

setup(
      ext_modules=cythonize("./rcit/cell_segment/cell_props.pyx"),
      include_dirs=[numpy.get_include()],)

setup(
      ext_modules=cythonize("./rcit/cell_segment/cell_segment_methods.pyx"),
      include_dirs=[numpy.get_include()],)

setup(
      ext_modules=cythonize("./rcit/util/post_process/motion_filter/motion_filter.pyx"),
      include_dirs=[numpy.get_include()],)

setup(
      ext_modules=cythonize("./rcit/motion/piv_motion.pyx"),
      include_dirs=[numpy.get_include()],)

# ext = [Extension("piv_motion",
#                  sources=["./rcit/motion/piv_motion.pyx"],
#                  extra_compile_args=["-fopenmp"],
#                  extra_link_args=["-fopenmp"])]

# setup(ext_modules=cythonize(ext, language_level=3))


setup(
      ext_modules=cythonize("./rcit/motion/vet_motion.pyx"),
      include_dirs=[numpy.get_include()],)
