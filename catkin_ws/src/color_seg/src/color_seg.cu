//Color based segmentation 

#include "utils.h"
#include <stdio.h>



//KERNEL PARA CONVERSIÓN A HSV
__global__
void rgba_to_greyscale(const uchar4* const rgbaImage,
                       uchar4* const greyImage,
                       int numRows, int numCols)
{ 

  int y = threadIdx.y+ blockIdx.y* blockDim.y;   //globalIdx = (blockIdx * threadsPerBlock) + threadId

  int x = threadIdx.x+ blockIdx.x* blockDim.x;
//printf("y= %d\n ",y);
// printf("x= %d\n ",x);
 // printf("threadIdx.y = %d\n",threadIdx.y);
 // printf("blockDim.y = %d\n",blockDim.y);
 // printf("blockIdx.y = %d\n",blockIdx.y);
 
if (y < numCols && x < numRows) {
  	int index = numRows*y +x;    ///numCols
/// printf("index = %d\n",index);
  uchar4 color = rgbaImage[index];
// printf("color.x = %u\n",(unsigned char)(color.x)); 
  unsigned char grey =  (unsigned char)(0.299f*color.x+ 0.587f*color.y + 0.114f*color.z);
  greyImage[index].x = grey ;
 greyImage[index].y =  grey;
 greyImage[index].z =  grey;
  }
}


void rgba2hsv(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
                            uchar4 * const d_greyImage, size_t numRows, size_t numCols)
{
  

  int   blockWidth = 32;   // (dimensionX / gridbloqueenX) = threadsporbloqueenX
  
   //  printf("numRows = %d\n",numRows);
   // printf("numCols = %d\n",numCols);

    const dim3 blockSize(blockWidth, blockWidth, 1);
   int   blocksX = (numRows/blockWidth)+1;       // +1 por redondeo ??
   int   blocksY = numCols/blockWidth +1; //TODO
   const dim3 gridSize( blocksX, blocksY, 1);  //TODO

///////////////

   rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
  cudaDeviceSynchronize(); 
  checkCudaErrors(cudaGetLastError());
}




