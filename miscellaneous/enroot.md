# ENROOT

```
export ENROOT_ROOT_PATH=PATH
export ENROOT_RUNTIME_PATH=$ENROOT_ROOT_PATH/runtime
export ENROOT_CACHE_PATH=$ENROOT_ROOT_PATH/tmp/enroot-cache/group-$(id -g)
export ENROOT_DATA_PATH=$ENROOT_ROOT_PATH/enroot-data/user-$(id -u)
export ENROOT_TEMP_PATH=$ENROOT_ROOT_PATH/temp
```

## init

```bash
enroot import docker://ubuntu
enroot create --name ubuntu ubuntu
enroot list
```

## start

```bash
enroot start --root --rw -m PATH_TO_MOUNT1 -m PATH_TO_MOUNT2 ubuntu
```
