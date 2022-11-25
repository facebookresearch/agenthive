# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import distutils.command.clean
import os
import shutil
import subprocess
from pathlib import Path

from setuptools import find_packages, setup

cwd = os.path.dirname(os.path.abspath(__file__))
version_txt = os.path.join(cwd, "version.txt")
with open(version_txt, "r") as f:
    version = f.readline().strip()


ROOT_DIR = Path(__file__).parent.resolve()


try:
    sha = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
        .decode("ascii")
        .strip()
    )
except Exception:
    sha = "Unknown"
package_name = "rlhive"

if os.getenv("BUILD_VERSION"):
    version = os.getenv("BUILD_VERSION")
elif sha != "Unknown":
    version += "+" + sha[:7]


def write_version_file():
    version_path = os.path.join(cwd, "rlhive", "version.py")
    with open(version_path, "w") as f:
        f.write("__version__ = '{}'\n".format(version))
        f.write("git_version = {}\n".format(repr(sha)))


def _get_pytorch_version():
    # if "PYTORCH_VERSION" in os.environ:
    #     return f"torch=={os.environ['PYTORCH_VERSION']}"
    return "torch"


def _get_packages():
    exclude = [
        "build*",
        "test*",
        # "rlhive.csrc*",
        # "third_party*",
        # "tools*",
    ]
    return find_packages(exclude=exclude)


ROOT_DIR = Path(__file__).parent.resolve()


class clean(distutils.command.clean.clean):
    def run(self):
        # Run default behavior first
        distutils.command.clean.clean.run(self)

        # Remove rlhive extension
        for path in (ROOT_DIR / "rlhive").glob("**/*.so"):
            print(f"removing '{path}'")
            path.unlink()
        # Remove build directory
        build_dirs = [
            ROOT_DIR / "build",
        ]
        for path in build_dirs:
            if path.exists():
                print(f"removing '{path}' (and everything under it)")
                shutil.rmtree(str(path), ignore_errors=True)


def _check_mj_envs():
    import importlib
    import sys

    name = "mj_envs"
    spam_loader = importlib.find_loader(name)
    found = spam_loader is not None

    if name in sys.modules:
        print(f"{name!r} already in sys.modules")
    # elif (spec := importlib.util.find_spec(name)) is not None:
    elif found:
        print(f"{name!r} is importable")
    else:
        raise ImportError(
            f"can't find {name!r}: check README.md for " f"install instructions"
        )


def _main():
    pytorch_package_dep = _get_pytorch_version()
    print("-- PyTorch dependency:", pytorch_package_dep)
    # branch = _run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    # tag = _run_cmd(["git", "describe", "--tags", "--exact-match", "@"])

    this_directory = Path(__file__).parent
    long_description = (this_directory / "README.md").read_text()

    # install mj_envs locally
    subprocess.run(
        [
            "git",
            "clone",
            "-c",
            "submodule.mj_envs/sims/neuromuscular_sim.update=none",
            "--branch",
            "v0.4dev",
            "--recursive",
            "https://github.com/vikashplus/mj_envs.git",
            "third_party/mj_envs",
        ]
    )
    subprocess.run(
        [
            "git",
            "clone",
            "--branch",
            "main",
            "https://github.com/pytorch/rl.git",
            "third_party/rl",
        ]
    )
    mj_env_path = os.path.join(os.getcwd(), "third_party", "mj_envs#egg=mj_envs")
    rl_path = os.path.join(os.getcwd(), "third_party", "rl#egg=torchrl")
    setup(
        # Metadata
        name="rlhive",
        version=version,
        author="rlhive contributors",
        author_email="vmoens@fb.com",
        url="https://github.com/fairinternal/rlhive",
        long_description=long_description,
        long_description_content_type="text/markdown",
        license="BSD",
        # Package info
        packages=find_packages(exclude=("test", "tutorials", "third_party")),
        # ext_modules=get_extensions(),
        # cmdclass={
        #     "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
        #     "clean": clean,
        # },
        install_requires=[
            pytorch_package_dep,
            # "torchrl @ git+ssh://git@github.com/pytorch/rl@main#egg=torchrl",
            f"torchrl @ file://{rl_path}",
            "gym==0.13",
            # "mj_envs",
            f"mj_envs @ file://{mj_env_path}",
            "numpy",
            "packaging",
            "cloudpickle",
            "hydra-core",
        ],
        zip_safe=False,
        dependency_links=[
            # location to your egg file
        ],
        extra_requires={
            "tests": ["pytest", "pyyaml", "pytest-instafail"],
        },
    )


if __name__ == "__main__":
    write_version_file()
    print("Building wheel {}-{}".format(package_name, version))
    print(f"BUILD_VERSION is {os.getenv('BUILD_VERSION')}")
    # _check_mj_envs()
    _main()
