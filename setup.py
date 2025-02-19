from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="coral-complexity-metrics",
    version="0.0.1",
    description="A Python project for estimating coral complexity metrics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Hannah White",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "appdirs",
        "packaging",
        "pymeshlab",
        "py",
        "pyparsing",
        "pytest",
        "six",
        "numpy",
        "pyvista",
        "tqdm"
    ],
    python_requires=">=3.7, <4",
)
