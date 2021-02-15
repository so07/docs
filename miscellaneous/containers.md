# Containers

## Singularity

Pull an image from a URI

```bash
singularity pull tf-20.11 docker://nvcr.io/nvidia/tensorflow:20.11-tf2-py3
```

launch a Singularity container with user-bind path specification
```bash
singularity shell -B PATH --nv tf-20.11
```

## ENROOT

https://github.com/NVIDIA/enroot

### config

```
export ENROOT_ROOT_PATH=PATH
export ENROOT_RUNTIME_PATH=$ENROOT_ROOT_PATH/runtime
export ENROOT_CACHE_PATH=$ENROOT_ROOT_PATH/tmp/enroot-cache/group-$(id -g)
export ENROOT_DATA_PATH=$ENROOT_ROOT_PATH/enroot-data/user-$(id -u)
export ENROOT_TEMP_PATH=$ENROOT_ROOT_PATH/temp
```

### init

Import an Ubuntu image from DockerHub

```bash
enroot import docker://ubuntu
enroot create --name ubuntu ubuntu
enroot list
```

### start

```bash
enroot start --root --rw -m PATH_TO_MOUNT1 -m PATH_TO_MOUNT2 ubuntu
```
