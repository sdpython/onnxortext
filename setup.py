# coding: utf-8
import os
import sys
import setuptools
import pathlib
import subprocess
from contextlib import contextmanager
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.develop import develop as _develop
from setuptools.command.build_py import build_py as _build_py
from setuptools import setup, find_packages
from pyquicksetup import read_version, read_readme, default_cmdclass


project_var_name = "onnxortext"
versionPython = "%s.%s" % (sys.version_info.major, sys.version_info.minor)
path = "Lib/site-packages/" + project_var_name
readme = 'README.rst'
history = "HISTORY.rst"
requirements = None

KEYWORDS = project_var_name + ', ONNX, onnxruntime'
DESCRIPTION = """Experimental runtime of ONNX operator based on onnxruntime."""
CLASSIFIERS = [
    'Programming Language :: Python :: %d' % sys.version_info[0],
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering',
    'Topic :: Education',
    'License :: OSI Approved :: MIT License',
    'Development Status :: 5 - Production/Stable'
]

packages = find_packages()
package_dir = {k: os.path.join('.', k.replace(".", "/")) for k in packages}
package_data = {"onnxortext": ["*.dll", "*.so"]}


TOP_DIR = os.path.dirname(__file__)


@contextmanager
def chdir(path):
    orig_path = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(orig_path)


def load_msvcvar():
    if os.environ.get('vcvars'):
        stdout, _ = subprocess.Popen([
            'cmd', '/q', '/c', '(%vcvars% & set)'],
            stdout=subprocess.PIPE, shell=True, universal_newlines=True).communicate()
        for line in stdout.splitlines():
            kv_pair = line.split('=')
            if len(kv_pair) == 2:
                os.environ[kv_pair[0]] = kv_pair[1]
    else:
        import shutil
        if shutil.which('cmake') is None:
            raise SystemExit(
                "Cannot find cmake in the executable path, "
                "please install one or specify the environement "
                "variable VCVARS to the path of VS vcvars64.bat.")


class BuildCMakeExt(_build_ext):

    def run(self):
        """
        Performs build_cmake before doing the 'normal' stuff
        """
        for extension in self.extensions:
            if extension.name == 'onnxortext._onnxortext':
                self.build_cmake(extension)

    def build_cmake(self, extension):
        import onnxruntime
        ortpath = os.path.abspath(os.path.dirname(onnxruntime.__file__))
        ortpath = os.path.normpath(os.path.join(ortpath, ".."))
        project_dir = pathlib.Path().absolute()
        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        ext_fullpath = pathlib.Path(self.get_ext_fullpath(extension.name))

        config = 'RelWithDebInfo' if self.debug else 'Release'
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=%s' % str(
                ext_fullpath.parent.absolute()),
            '-DONNXRUNTIME_LIB_DIR=%s' % ortpath,
            # '-DEXTENSION_NAME=%s' % pathlib.Path(
            #     self.get_ext_filename(extension.name)).name,
            '-DCMAKE_BUILD_TYPE=%s' % config
        ]

        # Uses to overwrite
        # export Python3_INCLUDE_DIRS=/opt/python/cp38-cp38
        # export Python3_LIBRARIES=/opt/python/cp38-cp38
        for env in ['Python3_INCLUDE_DIRS', 'Python3_LIBRARIES']:
            if env in os.environ:
                cmake_args.append("-D%s=%s" % (env, os.environ[env]))

        if self.debug:
            cmake_args += ['-DCC_OPTIMIZE=OFF']

        build_args = [
            '--config', config,
            '--parallel'
        ]

        with chdir(build_temp):
            self.spawn(['cmake', str(project_dir)] + cmake_args)
            if not self.dry_run:
                self.spawn(['cmake', '--build', '.'] + build_args)

        if sys.platform == "win32":
            self.copy_file(build_temp / config / 'onnxortext.dll',
                           os.path.dirname(self.get_ext_filename(extension.name)))
        else:
            self.copy_file(build_temp / config / 'onnxortext.so',
                           os.path.dirname(self.get_ext_filename(extension.name)))


class BuildPy(_build_py):
    def run(self):
        super().run()


class BuildDevelop(_develop):
    def run(self):
        super().run()


if sys.platform == "win32":
    load_msvcvar()


ext_modules = [
    setuptools.extension.Extension(
        name=str('onnxortext._onnxortext'),
        sources=[])]

cmd_class = default_cmdclass().copy()
cmd_class.update(dict(
    build_ext=BuildCMakeExt,
    build_py=BuildPy,
    develop=BuildDevelop))

setup(
    name='onnxortext',
    version=read_version(__file__, project_var_name),
    packages=packages,
    package_dir=package_dir,
    package_data=package_data,
    description="Experimental runtime of ONNX operator based on onnxruntime",
    license="MIT",
    author='Xavier Dupré',
    author_email='xavier.dupre@gmail.com',
    download_url='https://github.com/sdpython/onnxortext',
    url='http://www.xavierdupre.fr/app/onnxortext/helpsphinx/index.html',
    ext_modules=ext_modules,
    long_description=read_readme(__file__),
    keywords=KEYWORDS,
    cmdclass=cmd_class,
    classifiers=CLASSIFIERS,
    include_package_data=True,
    setup_requires=['pyquicksetup'],
    install_requires=['numpy', 'onnx', 'onnxruntime'],
)
