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


```shell
conda info
```

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

### Managing environments

####  Specifying a location for an environment

The location of an environment can be specified providing a path, with `--prefix`option, where the environment is created

```shell
conda create --prefix ./envs jupyterlab=0.35 matplotlib=3.1 numpy=1.16
```

in this way the name of the environment cannot be specified. To prevent long prefix in shell prompt the env_prompt setting in `.condarc` configuration file can be modified

```shell
conda config --set env_prompt '({name})'
```

#### Cloning an environment

A conda environment can be cloned and an exact copy of the environment is created

```shell
conda create --name cloned_env --clone env_to_clone
```

#### Building identical conda environments

In order to build identical conda environments an explicit list of all installed packages of an environment can be created

```shell
conda list --explicit
```

The list can be redirected on a file

```shell
conda list --explicit > spec-file.txt
```

The spec file can be used to create an identical environment

```shell
conda create --name myenv --file spec-file.txt
```

or to install all packages in an existing environment

```shell
conda install --name myenv --file spec-file.txt
```

#### View all packages in an environment

To see a list of all packages installed in a specific environment

```shell
conda list -n myenv
```

To see if a specific package is installed in an environment

```shell
conda list -n myenv scipy
```

```shell
conda info
```

####  Creating an environment from an environment.yml file

```shell
conda env create -f environment.yml
```


### conda channel

Conda channels are the locations where packages are stored. Conda packages are downloaded from remote channels, which are URLs to directories containing conda packages. The conda command searches a default set of channels and packages are automatically downloaded and updated from https://repo.anaconda.com/pkgs/. 

By default, conda prefers packages from a higher priority channel over any version from a lower priority channel. 

To add a new channel to the top of the channel list with the highest priority

```shell
conda config --add channels new_channel
```

or equivalent

```shell
conda config --prepend channels new_channel
```

To add a new channel to the bottom of the channel list with lowest priority

```shell
conda config --append channels new_channel
```

#### Creating custom channels

Channels are the path that conda takes to look for packages. The easiest way to use and manage custom channels is to use a private or public repository on Anaconda.org. 

If you do not wish to upload your packages to the Internet, you can build a custom repository served  locally using a `file://` URL.

First of all `conda-build` has to be installed

```shell
conda install conda-build
```

Then the packages must be organized in subdirectories with different platforms

```
channel/
linux-64/
 package-1.0-0.tar.bz2
linux-32/
 package-1.0-0.tar.bz2
osx-64/
 package-1.0-0.tar.bz2
win-64/
 package-1.0-0.tar.bz2
win-32/
 package-1.0-0.tar.bz2
```

Finally `conda index` command must be run on the channel root directory

```shell
conda index channel/
```


To check if a package is available on a channel 

```shell
conda search --override-channels --channel the_channel the_package
```


