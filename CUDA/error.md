# Error Handling in CUDA

```c
// macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {                                                             \
 cudaError_t err = cudaGetLastError();                                                 \
 if(err != cudaSuccess) {                                                              \
   printf("CUDA ERROR - %s %d : '%s'\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
   exit(0);                                                                            \
 }                                                                                     \
}
```
