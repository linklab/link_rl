import io
from setuptools import find_packages, setup
import pathlib

here = pathlib.Path(__file__).parent.resolve()


# Read in the README for the long description on PyPI
def long_description():
    with io.open('README.md', 'r', encoding='utf-8') as f:
        readme = f.read()
    return readme


setup(
    name='link_rl',
    version='0.1',
    description='reinforcement learning library and executables by LINK@KOREATECH',
    long_description=long_description(),
    url='https://github.com/linklab/link_rl.git',
    author='link.koreatech',
    author_email='link.koreatech@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'visdom==0.1.8.9'
    ],
    zip_safe=False
)