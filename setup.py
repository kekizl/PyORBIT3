from setuptools import Extension, setup
from pathlib import Path
import os
from os.path import join as pjoin
from distutils.command.build_ext import build_ext

cudaLocated = False
def find_in_path(name, path):
    """Find a file in a search path"""

    # Adapted fom http://code.activestate.com/recipes/52224
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None
def locate_cuda():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME env variable. If not found,
    everything is based on finding 'nvcc' in the PATH.
    """
       # First check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')

    else:
        # Otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, '
                'or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home': home, 'nvcc': nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    for k, v in iter(cudaconfig.items()):
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be '
                                   'located in %s' % (k, v))

    return cudaconfig


def customize_compiler_for_nvcc(self):
    """Inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on.
    """

    # Tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # Save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # Now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1
            # translated from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # Reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # Inject our redefined _compile method into the class
    self._compile = _compile



# Run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


try:
    CUDA = locate_cuda()
    cudaLocated = True
except EnvironmentError as e:
    print("CUDA could not be located:", e)
    print("Proceeding without CUDA support.")
# main dir is special
# we need it in include but not the actual main.cc
# libmain contains python package def, we don't want it in C++ sources

cuda_src = []
src = []

for f in Path("src").rglob("*.cu"):
    src.append(str(f))

for f in Path("src").rglob("*.cc"):
    excludes = ["main/main.cc"]
    include = True
    for e in excludes:
        if str(f).endswith(e):
            include = False
    if include:
        src.append(str(f))

include = []
for folder in os.walk("src"):
    excludes = ["src", "src/libmain", "src/libmain/orbit"]
    if folder[0] not in excludes:
        include.append(folder[0])
        print(folder[0])

#removed "-arch=sm_30" flag from extra_compile_args=because it throws errors
extension_mod = Extension(
    "orbit.core._orbit",
    sources=src,
    libraries=["fftw3"],
    include_dirs=include,
    extra_compile_args={
    'gcc': ["-DUSE_MPI=1", "-fPIC", "-lmpi", "-lmpicxx", "-Wl,--enable-new-dtags"],
    'nvcc': ["--ptxas-options=-v", "-c", "--compiler-options", "'-fPIC'"],
    },
    extra_link_args=["-lfftw3", "-lm", "-lmpi", "-lmpicxx", "-fPIC"],
)

if cudaLocated:
    extension_mod.sources = cuda_src + extension_mod.sources 
    extension_mod.library_dirs += CUDA['lib64']
    extension_mod.libraries += ["cudart"]
    extension_mod.runtime_library_dirs += CUDA['lib64']
    extension_mod.include_dirs += CUDA['include']
    extension_mod.extra_compile_args['nvcc'] += ["--compiler-options", "'-fPIC'"]
    extension_mod.extra_link_args += ["-lcudart", "-L/usr/local/cuda/lib64"]    
packages = ["orbit.core"]
for folder in os.walk("py/orbit"):
    path = os.path.normpath(folder[0])
    path = path.split(os.sep)
    packages.append(".".join(path[1:]))

package_dir = {
    "orbit": "py/orbit",
    "orbit.core": "src/libmain/orbit",
}

# This snippet generates the package structure of the orbit.core modules
# including the __init__.py file for each module
# The purpose is to be able to load individual modules from orbit.core in a
# Pythonic fashion.
core_modules = [
    "aperture",
    "orbit_mpi", 
    "trackerrk4",
    "error_base",
    "bunch",
    "teapot_base",
    "linac",
    "spacecharge",
    "orbit_utils",
    "foil",
    "collimator",
    "field_sources",
    "rfcavities",
    "impedances",
    "fieldtracker",
]

if cudaLocated:
    core_modules.append("orbit_cuda")

for mod in core_modules:
    packages.append(f"orbit.core.{mod}")
    package_dir.update({f"orbit.core.{mod}": "src/libmain/module_template"})

# Define the setup parameters
setup(
    ext_modules=[extension_mod],
    package_dir=package_dir,
    packages=packages,
    #inject custom trigger
    cmdclass={'build_ext': custom_build_ext},
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    scripts=["bin/pyORBIT"],
)
