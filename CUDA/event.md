# Simple example of events in CUDA

```c
cudaEvent_t start, stop;
 
cudaEventCreate(&start);
cudaEventCreate(&stop);
 
cudaEventRecord(start);
 
kernel<<<grid, block>>>(...);
 
cudaEventRecord(stop);
 
cudaEventSynchronize(stop);

float Etime; // Elapsed time in ms
cudaEventElapsedTime(&Etime, start, stop);
 
cudaEventDestroy(start);
cudaEventDestroy(stop);

```
