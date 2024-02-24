from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name='paref',
    version='0.2.0',
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
        'Development Status :: 3 - Alpha'
    ],
    python_requires='>=3.10',
    install_requires=[
        'numpy',
        'scipy',
        'plotly',
        'gpytorch',
        'torch',
        'matplotlib',
        'tqdm',
        'tabulate',
    ],
    extras_require={
        'dev': ['pre-commit', 'flake8', 'flake8-print'],
        'functional_tests': ['ipython', 'jupyter', 'pymoo'],
        'test': ['pytest', 'pytest-cov'],
    },
    test_suite='tests',
    packages=find_packages(
        exclude=['dashboard*', 'functional_tests'],  # alternatively: `exclude=['additional*']`
    ),
)
