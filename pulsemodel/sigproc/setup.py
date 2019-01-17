'''
Copyright(C) 2016 Engineering Department, University of Cambridge, UK.

License
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

Author
    Gilles Degottex <gad27@cam.ac.uk>
'''

# Profile:
#python -m cProfile -s cumulative test_sample.py > test_profile.txt

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("sinusoidal", ["cython/sinusoidal.pyx"],
    include_dirs=[numpy.get_include()],
    extra_compile_args=['-Wno-cpp']
    ),
]

setup(
    name="sigproc",
    url="https://github.com/gillesdegottex/sigproc",
    author_email='gad27@cam.ac.uk',
    ext_modules = cythonize(extensions)
)

