from setuptools import setup, find_packages

setup(
    name='PytorchFS',
    version='1.1.1',
    description='Pytorch Fault Simulator',
    author='4thMemorize',
    author_email='woobin.ko@yonsei.ac.kr',
    url='https://github.com/4thMemorize/Pytorch-Fault-Simulator',
    install_requires=['torch', 'numpy'],
    packages=find_packages(exclude=[]),
    keywords=['pytorchFS', '4thMemorize', 'python fault', 'pytorch fault', 'pytorch fault injection', 'pytorch fault simulator'],
    python_requires='>=3.6',
    package_data={},
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
