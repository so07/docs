# Python environments


## Virtual environment

A virtual environment is a self confinement Python packages installation that can coexist with other installations on the same system.
Thus an application can not share dependency with other application.

### virtual environment basic usage

1. Create a virtual environment for a project
1. Activate virtual environment
1. Install packages and/or Execute commands
1. When done deactivate virtual environment

#### Create a virtual environment

```shell
python3 -m venv her_env
```

By default `venv` module does not include global Python site-packages directory.
To create a virtual environment including global site-packages directory:

```shell
python -m venv --system-site-packages my_venv
```

#### Activate virtual environment

```shell
source her_env/bin/activate
```

#### Deactivate a virtual environment

```shell
deactivate
```

### The requirement file

If you want to specify all dependencies and versions used in a project you can use a requirement file.
A requirement file allows to specify which packages and versions should be installed for the project.

```shell
pip freeze > requirements.txt
```

Once you have a requirement file you can replicate the same environment in another system with the command `pip install` with `-r` option:
```shell
pip install -r requirements.txt
```

## Conda

To see a list of all of your environments
```shell
conda info --envs
```
or
```shell
conda env list
```
your current environment is highlighted with an asterisk

To create a conda environment
```shell
conda create --name her_conda
```

```shell
conda create --name her_conda python=3.5
```

To activate a conda environment
```shell
conda activate her_conda
```

To deactivate a conda environment
```shell
conda deactivate
```

```shell
conda install anaconda-clean
anaconda-clean
```

To remove an environment
```shell
conda env remove -n ENV_NAME
```
or
```shell
conda remove --name ENV_NAME --all
```

NB: To prevent Conda from activating the base environment by default

```shell
conda config --set auto_activate_base false
```

Use the --prefix or -p option to specify where to write the environment files
```shell
conda create --prefix PATH
```

### Using pip in an environment

To use pip in your environment
```shell
conda install -n myenv pip
conda activate myenv
```
