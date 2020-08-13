import pathlib
from setuptools import setup

# The directory containing this file
root_path = pathlib.Path(__file__).parent
long_description = (root_path / "README.rst").read_text()

setup(
    name='GDEFReader',
    version='0.0.1a13',
    packages=['gdef_reader'],
    url='https://github.com/natter1/gdef_reader',
    license='MIT',
    author='Nathanael Jöhrmann',
    author_email='',
    description='Tool to read/process *.gdf AFM measurement files',
    long_description=long_description,
    install_requires=['matplotlib', 'numpy', 'scipy', 'natsort'],
)
# python-pptx-interface