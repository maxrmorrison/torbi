from setuptools import setup


with open('README.md', encoding='utf-8') as file:
    long_description = file.read()


setup(
    name='torbi',
    description='Viterbi decoding in PyTorch',
    version='0.0.1',
    author='Max Morrison',
    author_email='maxrmorrison@gmail.com',
    url='https://github.com/maxrmorrison/torbi',
    install_requires=['torch'],
    packages=['torbi'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['decode', 'sequence', 'torch', 'Viterbi'],
    classifiers=['License :: OSI Approved :: MIT License'],
    license='MIT')
