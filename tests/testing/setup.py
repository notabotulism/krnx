"""
Chillbot - AI Memory Infrastructure
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="chillbot",
    version="0.1.0",
    author="Ben",
    author_email="ben@chillbot.io",
    description="AI Memory Infrastructure - The substrate for AI agents that remember",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chillbot/chillbot",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Database",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=[
        "httpx>=0.25.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "compute": [
            "sentence-transformers>=2.2.0",
            "qdrant-client>=1.7.0",
        ],
        "server": [
            "fastapi>=0.109.0",
            "uvicorn>=0.27.0",
            "redis>=5.0.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.23.0",
            "black>=24.0.0",
            "ruff>=0.1.0",
        ],
        "all": [
            "sentence-transformers>=2.2.0",
            "qdrant-client>=1.7.0",
            "fastapi>=0.109.0",
            "uvicorn>=0.27.0",
            "redis>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "chillbot-server=chillbot.server:main",
            "chillbot-worker=chillbot.compute.worker:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/chillbot/chillbot/issues",
        "Source": "https://github.com/chillbot/chillbot",
        "Documentation": "https://docs.chillbot.io",
    },
)
