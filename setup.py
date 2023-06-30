from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name='paref',
    version='0.1.0',
    license='MIT',
    description='Pareto reflection based multi-objective optimization',
    long_description=readme(),
    long_description_content_type='text/markdown',
    author='Nicolai Palm',
    author_email='nicolaipalm@googlemail.com',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'scipy',
        'plotly',
        'gpytorch',
        'torch',
        'matplotlib',
        'tqdm',
    ],
    extras_require={
        'dev': ['pre-commit', 'flake8', 'flake8-print'],
        'examples_': ['ipython', 'jupyter', 'pymoo'],
        'test': ['pytest', 'pytest-cov'],  # "nbval"],
    },
    test_suite='tests',
)
