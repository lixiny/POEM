from setuptools import setup, find_packages

setup(
    name="POEM",
    version="0.0.1",
    python_requires=">=3.8.0",
    packages=find_packages(exclude=("manotorch", "assets", "checkpoints", "venvs", "thirdparty"
                                    "common", "config", "data", "exp", "scripts", "tmp", "docs")),
)
