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
        'numpy; python_version~="1.21.2"',
        'scipy; python_version~="1.7.1"',
        'pymoo; python_version~="0.5.0"',
        'plotly; python_version~="5.1.0"',
        'gpytorch; python_version~="1.5.1"',
        'torch; python_version~="1.12.0"',
        'matplotlib; python_version~="3.4.2"',
        'tqdm; python_version~="4.62.2"',
    ],

)
