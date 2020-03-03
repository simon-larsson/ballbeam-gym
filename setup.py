from setuptools import setup, find_packages
import ballbeam_gym

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='ballbeam_gym',
    version=ballbeam_gym.__version__,
    description='Ball & beam environments for OpenAI gym',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    author='Simon Larsson',
    author_email='simonlarsson0@gmail.com',
    url='https://github.com/simon-larsson/ballbeam-gym',
    license='MIT',
    install_requires=['gym', 'numpy', 'matplotlib'],
    classifiers=['License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 3 :: Only',
                 'Topic :: Scientific/Engineering']
)
