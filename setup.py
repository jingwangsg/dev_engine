from setuptools import setup, find_packages
import site
import shutil
import os

def copy_file(source, destination):
    if os.path.exists(source):
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        os.system(f"rm -rf {destination}")
        os.system(f"cp -v {os.path.abspath(source)} {destination}")

def copy_debug_script():
    print("Copying debugpy_cli.py")
    source = "dev_engine/debug/debugpy_utils.py"
    destination = os.path.join(site.getsitepackages()[0], "debugpy_cli.py")
    copy_file(source, destination)

def copy_bkpt_script():
    print("Copying bkpt.py")
    source = "dev_engine/debug/bkpt.py"
    destination = os.path.join(site.getsitepackages()[0], "bkpt.py")
    copy_file(source, destination)

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="dev_engine",
    version="0.1.0",
    description="A development engine for scientific AI workflows",
    author="jing",
    author_email="jing005@e.ntu.edu.sg",
    license="MIT",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests", "docs*", "scripts*", "*checkpoints*"]),
    include_package_data=True,
    package_data={"dev_engine": ["py.typed"]},
    install_requires=requirements,
)

copy_debug_script()
copy_bkpt_script()