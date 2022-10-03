"""setup.py for project 'epic'"""

from setuptools import setup, find_packages


def get_readme() -> str:
    """Retrieve content from README."""
    with open("README.md", "r", encoding="utf-8") as f:
        return f.read()


IMITATION_REQUIRE = [
    'imitation @ git+https://github.com/HumanCompatibleAI/imitation.git@master',
    "stable_baselines3",
    "seals @ git+https://github.com/HumanCompatibleAI/seals.git@master",
]

TESTS_REQUIRE = [
    "pytest",
    "pytest-cov",
    "coverage",
    "codecov",
    "mypy",
    "pytype",
    "flake8",
    "pylint",
    "black",
    *IMITATION_REQUIRE,
]


DEV_REQUIRE = [
    "isort",
    "pre-commit",
    *TESTS_REQUIRE,
]

setup(
    name='epic',
    version='0.1.0',
    description='Epic implements the Equivalent-Policy Invariant Comparison (EPIC) '
                'distance for reward functions.',
    long_description=get_readme(),
    long_description_content_type='text/markdown',
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"imitation": ["py.typed"]},
    python_requires='>=3.9',
    install_requires=[
        'numpy',
        'torch',
        ],
    tests_require=TESTS_REQUIRE,
    extras_require={
        "dev": DEV_REQUIRE,
        "test": TESTS_REQUIRE,
        "imitation": IMITATION_REQUIRE,
    },
    url='https://github.com/HumanCompatibleAI/epic',
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)