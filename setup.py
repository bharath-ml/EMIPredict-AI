"""
Setup configuration for EMIPredict-AI package
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="emipredict-ai",
    version="2.0.0",
    author="EMIPredict AI Team",
    author_email="contact@emipredict.ai",
    description="Intelligent Financial Risk Assessment Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/EMIPredict-AI",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "emipredict=app:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/EMIPredict-AI/issues",
        "Source": "https://github.com/yourusername/EMIPredict-AI",
        "Documentation": "https://emipredict-ai.readthedocs.io/",
    },
)