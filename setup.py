# import multiprocessing to avoid this bug (http://bugs.python.org/issue15881#msg170215)
import multiprocessing

assert multiprocessing
import re
from setuptools import setup, find_packages


def get_version():
    """
    Extracts the version number from the version.py file.
    """
    VERSION_FILE = "easier/version.py"
    mo = re.search(
        r'^__version__ = [\'"]([^\'"]*)[\'"]', open(VERSION_FILE, "rt").read(), re.M
    )
    if mo:
        return mo.group(1)
    else:
        raise RuntimeError("Unable to find version string in {0}.".format(VERSION_FILE))


install_requires = ["click"]

tests_require = [
    "coverage",
    "flake8",
    "pytest",
]

extras_require = {"dev": tests_require}

setup(
    name="easier",
    version=get_version(),
    description="Tools for analysis",
    long_description="Tools for analysis",
    url="https://github.com/robdmc/easier",
    author="Rob deCarvalho",
    author_email="unlisted@unlisted.net",
    keywords="",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT",
    include_package_data=True,
    test_suite="nose.collector",
    install_requires=install_requires,
    tests_require=tests_require,
    extras_require=extras_require,
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "ezr.gsheet = easier.cli:gsheet",
            "ezr.gsheet_push = easier.cli:gsheet_push",
            "ezr.sfdc = easier.cli:sfdc",
        ],
    },
)
