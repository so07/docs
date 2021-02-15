# Containers

## Singularity

### Download pre-built images

Pull an image from a URI

```bash
singularity pull --name tf-20.11 docker://nvcr.io/nvidia/tensorflow:20.11-tf2-py3
```

### 

#### ```shell```

The ```shell``` command allows you to spawn a new shell within your container and interact with it as though it were a small virtual machine.

Launch a Singularity container with user-bind path specification
```bash
singularity shell -B PATH --nv tf-20.11
```

#### ```exec```

The ```exec``` command allows you to execute a custom command within a container by specifying the image file

```bash
singularity exec tf-20.11 ls
```

#### ```run```

Launch a Singularity container and execute a runscript if one is defined for that container.

```bash
singularity run tf-20.11
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
