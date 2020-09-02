# NVTX usage in CUDA

```cmake
# This should be added AFTER the FindCUDA macro has been run
IF(USE_NVTX)
  IF(HAVE_CUDA)
    ADD_DEFINITIONS(-DUSE_NVTX)
    LINK_DIRECTORIES("${CUDA_TOOLKIT_ROOT_DIR}/lib64")
    LINK_LIBRARIES("nvToolsExt")
  ENDIF(HAVE_CUDA)
ENDIF(USE_NVTX)
```
