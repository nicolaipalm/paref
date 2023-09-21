from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name='paref',
    version='0.1.8',
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
    python_requires='>=3.9',
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
        'examples': ['ipython', 'jupyter', 'pymoo'],
        'test': ['pytest', 'pytest-cov'],
    },
    test_suite='tests',
)
