from setuptools import setup

setup(
    name='paref',
    version='1.0.0',
    license='MIT',
    description='Small package containing implementations and interfaces for weight-based multi-objective '
                'optimization approaches',
    author='Nicolai Palm',
    author_email='nicolaipalm@googlemail.com',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status:: 1 - Planning'
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'scipy',
        'pymoo',
        'plotly',
        'gpytorch',
        'torch',
        'matplotlib',
        'tqdm',
    ],
    extras_require={
        'dev': ['pre-commit', 'flake8', 'flake8-print'],
        'examples': ['ipython', 'jupyter'],
        'test': ['pytest', 'pytest-cov'],  # "nbval"],
    },
    test_suite='tests',
)
