try:
    from setuptools import setup
    setup
except ImportError:
    from distutils.core import setup
    setup


setup(
    name='uvmod',
    version='0.2',
    author='Ilya Pashchenko',
    author_email='in4pashchenko@gmail.com',
    packages=['uvmod'],
    scripts=['bin/fit_amp.py'],
    url='',
    test_suit='tests',
    license='LICENSE',
    description='Simple models',
    long_description=open('README.rst').read(),
    install_requires=[
        "numpy >= 1.7.2",
        "scipy >= 0.12.0",
        "emcee >= 1.2.0"
    ],)
