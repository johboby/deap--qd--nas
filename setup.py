"""
DEAP多目标优化框架安装配置
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="deap-moo-framework",
    version="2.0.0",
    author="DEAP Research Team",
    author_email="research@deap.org",
    description="现代化多目标优化研究框架",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/deap/deap-moo",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries",
    ],
    python_requires=">=3.8",
    install_requires=[
        "deap>=1.3.1",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "ml": ["scikit-learn", "pandas"],
        "parallel": ["pathos"],
        "viz": ["plotly"],
        "dev": ["pytest", "black", "flake8"],
    },
    entry_points={
        "console_scripts": [
            "deap-demo=src.main_clean:demo_basic_usage",
        ],
    },
)