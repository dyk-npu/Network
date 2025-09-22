from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="cad-feature-recognition",
    version="1.0.0",
    author="CAD Research Team",
    author_email="contact@cad-research.com",
    description="Multi-modal deep learning system for CAD feature recognition, classification and retrieval",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/cad-feature-recognition",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Multimedia :: Graphics :: 3D Modeling",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.1.0",
            "pytest-cov>=3.0.0",
            "black>=22.6.0",
            "flake8>=5.0.0",
            "isort>=5.10.0",
            "mypy>=0.971",
        ],
        "gpu": [
            "cupy>=11.0.0",
            "pytorch3d>=0.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cad-train=scripts.train:main",
            "cad-evaluate=scripts.evaluate:main",
            "cad-inference=scripts.inference:main",
            "cad-preprocess=scripts.preprocess_data:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)