"""
FloodML - Machine Learning for Flood Prediction
A Python package for predicting flood events using USGS streamflow and NWS weather data
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="floodml",
    version="0.1.0",
    author="FloodML Contributors",
    author_email="info@floodml.org",
    description="Machine Learning for Flood Prediction using USGS and NWS data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/floodml",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/floodml/issues",
        "Documentation": "https://floodml.readthedocs.io/",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Hydrology",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
        "interactive": [
            "jupyter>=1.0",
            "plotly>=5.0",
            "bokeh>=2.4",
        ],
    },
    entry_points={
        "console_scripts": [
            "floodml-predict=floodml.cli:predict",
            "floodml-train=floodml.cli:train",
        ],
    },
    include_package_data=True,
    package_data={
        "floodml": [
            "data/*.json",
            "config/*.yaml",
        ],
    },
)