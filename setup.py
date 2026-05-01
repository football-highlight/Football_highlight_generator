from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="football-highlights-generator",
    version="1.0.0",
    author="Football Highlights Team",
    author_email="arrbean1810@gmail.com",
    description="Real-time football match highlights generation using multi-modal ML",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Arbin1810/football-highlights",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "football-highlights=src.main_pipeline:main",
            "fhl-train=scripts.train_model:main",
            "fhl-process=scripts.preprocess_data:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.txt"],
    },
)