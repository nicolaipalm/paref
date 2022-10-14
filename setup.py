from setuptools import setup

setup(
    name='weimoo',
    version='1.0.0',
    license='MIT',
    description='Small package containing implementations and interfaces for weight-based multi-objective optimization approaches',
    author='Nicolai Palm',
    author_email='nicolaipalm@googlemail.com',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status:: 1 - Planning'
    ],
    package_dir={'': 'src'},
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

)
