# Packaging a python project


#### https://blog.ionelmc.ro/2014/05/25/python-packaging/

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

```
from setuptools import setup, find_packages

setup(
    name="HerProject",
    version="1.0",
    author="so07",
    author_email="so07@mail.com",
    description="A simple description of HerProject project",
    long_description="""A longer description of HerProject project.
                        more detaile here""",
    url="https://her_project_url.com",
    packages=find_packages(),
)
```


```
from setuptools import setup, find_packages

setup(
    name="HerProject",
    version="1.0",
    author="so07",
    author_email="so07@mail.com",
    description="A simple description of HerProject project",
    long_description="""A longer description of HerProject project.
                        more detaile here""",
    url="https://her_project_url.com",
    packages=find_packages("src"),
    package_dir={"": "src"},
)
```

```
from setuptools import setup, find_packages

setup(
    name="HerProject",
    version="1.0",
    author="so07",
    author_email="so07@mail.com",
    description="A simple description of HerProject project",
    long_description="""A longer description of HerProject project.
                        more detaile here""",
    url="https://her_project_url.com",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=["required_package"],
)
```

Python packages can include command line tools.
A common approach is to use ```entry_point``` feature of ```setuptools```.

A very useful type of entry point is ```console_script``` that allows Python functions to be registered as command-line executable.

```
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

the complete example
```
from setuptools import setup, find_packages

setup(
    name="HerProject",
    version="1.0",
    author="so07",
    author_email="so07@mail.com",
    description="A simple description of HerProject project",
    long_description="""A longer description of HerProject project.
                        more detaile here""",
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

```
python setup.py register
```

```
python setup.py sdist
```

```
pip install twine
```

```
twine upload dist/*
```


