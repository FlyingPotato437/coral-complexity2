try:
    from setuptools import setup, find_packages
    import sys
    from setuptools.command.test import test as TestCommand
except ImportError:
    from distutils.core import setup

with open('README.md') as f:
    readme = f.read()

with open('LICENSE.md') as f:
    license = f.read()

class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def run_tests(self):
        import shlex
        #import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)

setup(
    description='Package to measure rugosity of 3D meshes',
    long_description=readme,
    author='Steve Hawes',
    url="https://github.com/shawes/mesh3d-python",
    author_email="maxhawes@gmail.com",
    install_requires=['pytest'],
    tests_require=['pytest'],
    cmdclass={'test': PyTest},
    scripts=[],
    name='mesh3d',
    version="1.0.1",
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
