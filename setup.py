from distutils.core import setup
# from Cython.Build import cythonize
# from Cython.Distutils import Extension
# import numpy as np

# ext = Extension(name="coordinate_descent" ,sources=["kcsd/cythonized/coordinate_descent.pyx"], libraries=["gsl", "gslcblas"])

setup(name="hearth",
      version="0.0.1",
      packages=["hearth"],
      install_requires=['numpy', 'pandas']),
      # ext_modules=cythonize(ext),
      # include_dirs=[np.get_include()])