"""
Setup configuration for EcoRRAP: Enhanced Coral Reef Complexity Metrics
"""

from setuptools import setup, find_packages

setup(
    name="coral-complexity-metrics",
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
)
