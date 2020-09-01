# Packaging a python project

## Packaging structure

```
├─ src
│  └─ packagename
│     ├─ __init__.py
│     └─ ...
├─ tests
│  └─ ...
└─ setup.py
```


## ```setup.py``` file

In order to build, install and distribuite Python package the ```setuptools``` is commonly used. \
First of all we have to produce a ```setup.py``` file and place it in the root directory of the project. \
The ```setuptools``` package provides a function ```setup``` to build and install your package.



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

In the case of project that use ```src``` directory as the root directory of source codes the first argument of find_package is ```src``` directory.

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

```setuptools``` supports automatically installing dependency when building package defining ```install_requires``` argument to specify the dependency and the Python packages required.

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

Python packages can include command line tools.
A common approach is to use ```entry_point``` feature of ```setuptools```.

A very useful type of entry point is ```console_script``` that allows Python functions to be registered as command-line executable.

```python
from setuptools import setup, find_packages

setup(
    ...
    entry_points = {
        "console_scripts": [
            "name=her_project.file:main'
        ],
    }
    ...
)
```

the complete example ```setup.py``` file

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
    entry_points = {
        "console_scripts": [
            "name=her_project.file:main'
        ],
    }
)
```

## Publish on ```PyPi```

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

- https://blog.ionelmc.ro/2014/05/25/python-packaging/

