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

```shell
conda info --envs
```

```shell
conda create --name her_conda
```

```shell
conda create --name her_conda python=3.5
```

```shell
conda activate her_conda
```

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

NB: To prevent Conda from activating the base environment by default

```shell
conda config --set auto_activate_base false
```
