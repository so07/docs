# Python environments


## Virtualenvironment

```shell
python3 -m venv her_env
```

```shell
source her_env/bin/activate
```

```shell
deactivate
```

```shell
pip freeze > requirements.txt
```

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

## Using pip in an environment

To use pip in your environment
```shell
conda install -n myenv pip
conda activate myenv
```
