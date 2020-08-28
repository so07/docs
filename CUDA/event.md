# Simple example of events in CUDA

```cpp
cudaEvent_t start, stop;
 
// create events
cudaEventCreate(&start);
cudaEventCreate(&stop);
 
// start recording
cudaEventRecord(start);

// ...
 
// end recording
cudaEventRecord(stop);
 
// synchronize on event stop
cudaEventSynchronize(stop);

float Etime; // Elapsed time in ms
cudaEventElapsedTime(&Etime, start, stop);
 
// release event resources
cudaEventDestroy(start);
cudaEventDestroy(stop);

```

The following is a complete example of measuring the Host to Device bandwith of memory transfers

```cpp
#include <iostream>
#include <cuda_runtime.h>

int main () {

   int num_elemnts = 1<<8;

   // allocate memory
   int * h_buffer = (int *) malloc(sizeof(int) * num_elemnts);

   int * d_buffer;
   cudaMalloc((void **) &d_buffer, sizeof(int) * num_elemnts);

   cudaEvent_t start, stop;

   // create events
   cudaEventCreate(&start);
   cudaEventCreate(&stop);

   // start recording
   cudaEventRecord(start);

   cudaMemcpy(h_buffer, d_buffer, sizeof(int) * num_elemnts, cudaMemcpyHostToDevice);

   // end recording
   cudaEventRecord(stop);

   // synchronize on event stop
   cudaEventSynchronize(stop);

   float Etime; // Elapsed time in ms
   cudaEventElapsedTime(&Etime, start, stop);

   // release event resources
   cudaEventDestroy(start);
   cudaEventDestroy(stop);

   std::cout << "Etime / ms = " << Etime << std::endl;

   return 0;
}

```
