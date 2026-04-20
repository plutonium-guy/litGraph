"""PEP 561 type stub package for litGraph.

Install via:
    pip install ./litgraph-stubs

Then Pyright/mypy/Pylance will discover types for `litgraph` and its submodules.
"""
from setuptools import setup

setup(
    name="litgraph-stubs",
    version="0.1.0",
    description="Type stubs for litgraph",
    packages=["litgraph-stubs"],
    package_data={
        "litgraph-stubs": ["*.pyi", "py.typed"],
    },
    include_package_data=True,
    python_requires=">=3.9",
)
