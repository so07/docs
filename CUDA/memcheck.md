# [`cuda-memcheck`](https://docs.nvidia.com/cuda/cuda-memcheck/index.html)

`cuda-memcheck` tool is able to detect and attribute out of bounds and misaligned memory access errors in CUDA applications.
It is installed as part of the CUDA toolkit.

`cuda-memcheck` tool do not need any special compilation flags to function.
The output is more useful with some extra compiler flags.
The `-G` option to nvcc forces the compiler to generate debug information for the CUDA application.
To generate line number information for applications without affecting the optimization level of the output, the `-lineinfo` option to nvcc can be used.

The `cuda-memcheck` tool can fail to initialize when there are a lot of CUDA functions in the target app.
The environment variable `CUDA_MEMCHECK_PATCH_MODULE` can be set to 1 in order to bypass this behavior, thus resolving the initialization error.


## usage

```
cuda-memcheck [memcheck_options] app_name [app_options]
```

## usage with MPI

```
mpirun -np 2 cuda-memcheck ./myapp <args>
```

Useful script to avoid interleaved output of different processes:
```
#!/bin/bash
LOG=$1.$OMPI_COMM_WORLD_RANK
#LOG=$1.$MV2_COMM_WORLD_RANK

export COMPUTE_PROFILE=0
export CUDA_MEMCHECK_PATCH_MODULE=1

cuda-memcheck --log-file $LOG.log --save $LOG.memcheck $*
```

Run memcheck script as:
```
mpiexec -np 2 cuda-memcheck-script.sh ./myapp <args>
```

## read output

```
cuda-memcheck --read <ouput>
```
