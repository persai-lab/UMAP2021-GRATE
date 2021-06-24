from setuptools import setup
from setuptools import find_packages

setup(
    name="GRATE",
    version="1.0",
    description="Granular Rank-based Tensor Factorization for Knowledge Tracing and Modeling",
    download_url="",
    license="MIT",
    install_requires=["numpy", "scikit-learn", "scikit-surprise"],
    include_package_data=False,
    packages=find_packages("GRATE", exclude=("results", "data"))
)
