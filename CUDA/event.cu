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
