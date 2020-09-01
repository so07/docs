# Packaging a python project

## Packaging structure

The src directory structure is preferable
```
├─ src
│  └─ packagename
│     ├─ __init__.py
│     └─ ...
├─ tests
│  └─ ...
└─ setup.py
```

The `src` directory is a better layout rather than the following

```
├─ packagename
│  ├─ __init__.py
│  └─ ...
├─ tests
│  └─ ...
└─ setup.py
```

## ```setup.py``` file

In order to build, install and distribuite Python package the `setuptools` is commonly used. \
First of all we have to produce a `setup.py` file and place it in the root directory of the project. \
The `setuptools` package provides a function `setup` to build and install your package.


```python
from setuptools import setup, find_packages

setup(
    name="HerProject",
    version="1.0",
    author="so07",
    author_email="so07@mail.com",
    description="A simple description of HerProject project",
    long_description="""A longer description of HerProject project.
                        more details here""",
    url="https://her_project_url.com",
    packages=find_packages(),
)
```

In the case of project that use `src` directory as the root directory of source codes the first argument of find_package is `src` directory.

```python
from setuptools import setup, find_packages

setup(
    name="HerProject",
    version="1.0",
    author="so07",
    author_email="so07@mail.com",
    description="A simple description of HerProject project",
    long_description="""A longer description of HerProject project.
                        more details here""",
    url="https://her_project_url.com",
    packages=find_packages("src"),
    package_dir={"": "src"},
)
```

### How to install required packages

`setuptools` supports automatically installing dependency when building package defining `install_requires` argument to specify the dependency and the Python packages required.

```python
from setuptools import setup, find_packages

setup(
    name="HerProject",
    version="1.0",
    author="so07",
    author_email="so07@mail.com",
    description="A simple description of HerProject project",
    long_description="""A longer description of HerProject project.
                        more details here""",
    url="https://her_project_url.com",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=["required_package"],
)
```
### Command line tools

Python packages can include command line tools.
A common approach is to use `entry_point` feature of `setuptools`.

A very useful type of entry point is `console_script` that allows Python functions to be registered as command-line executable.

```python
from setuptools import setup, find_packages

setup(
    ...
    entry_points = {
        "console_scripts": [
            "name=her_project.file:main"
        ],
    }
    ...
)
```

### README file

Reading `long_description` entry from `README.md` file in markdown syntax

```python
from setuptools import setup, find_packages

# read the contents of your README file
from os import path
with open(path.join(path.abspath(path.dirname(__file__)), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    ...
    long_description=long_description,
    ...
)
```

more complex case reading from `README.rst` file in rst format
(see [here](https://blog.ionelmc.ro/2014/05/25/python-packaging/) for source)

```python
from setuptools import setup, find_packages

iport re
import io
from os import path

def read(*names, **kwargs):
    with io.open(
        path.join(path.dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ) as fh:
        return fh.read()

setup(
    ...
    long_description='%s\n%s' % (
        re.compile('^.. start-badges.*^.. end-badges', re.M | re.S).sub('', read('README.rst')),
        re.sub(':[a-z]+:`~?(.*?)`', r'``\1``', read('CHANGELOG.rst'))
    ),
    ...
)
```
### `PyPi` classifiers

```python
setup(
    ...
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Utilities',
    ],
    ...
)
```

### Other requirements

```python

setup(
    ...
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*',
    install_requires=[
        # eg: 'aspectlib==1.1.1', 'six>=1.7',
    ],
    extras_require={
        # eg:
        #   'rst': ['docutils>=0.11'],
        #   ':python_version=="2.6"': ['argparse'],
    },
    setup_requires=[
        'pytest-runner',
    ],
)

```

### The complete `setup.py` file

the complete example `setup.py` file

```python
from setuptools import setup, find_packages

# read the contents of your README file
from os import path
with open(path.join(path.abspath(path.dirname(__file__)), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="HerProject",
    version="1.0",
    author="so07",
    author_email="so07@mail.com",
    description="A simple description of HerProject project",
    long_description=long_description,
    url="https://her_project_url.com",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        # eg: 'aspectlib==1.1.1', 'six>=1.7',
    ],
    extras_require={
        # eg:
        #   'rst': ['docutils>=0.11'],
        #   ':python_version=="2.6"': ['argparse'],
    },
    setup_requires=[
        #eg: 'pytest-runner',
    ],
    entry_points = {
        "console_scripts": [
            "name=her_project.file:main",
        ],
    },
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Utilities',
    ],
)
```

## Publish on `PyPI`

```shell
python setup.py register
```

```shell
python setup.py sdist
```

```shell
pip install twine
```

```shell
twine upload dist/*
```


## References

- https://gitlab.hpc.cineca.it/scai-training-rome/python-scientific
- https://blog.ionelmc.ro/2014/05/25/python-packaging/

