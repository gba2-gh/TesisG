// Homework 1
// Color to Greyscale Conversion

//A common way to represent color images is known as RGBA - the color
//is specified by how much Red, Green, and Blue is in it.
//The 'A' stands for Alpha and is used for transparency; it will be
//ignored in this homework.

//Each channel Red, Blue, Green, and Alpha is represented by one byte.
//Since we are using one byte for each color there are 256 different
//possible values for each color.  This means we use 4 bytes per pixel.

//Greyscale images are represented by a single intensity value per pixel
//which is one byte in size.

//To convert an image from color to grayscale one simple method is to
//set the intensity to the average of the RGB channels.  But we will
//use a more sophisticated method that takes into account how the eye 
//perceives color and weights the channels unequally.

//The eye responds most strongly to green followed by red and then blue.
//The NTSC (National Television System Committee) recommends the following
//formula for color to greyscale conversion:

//I = .299f * R + .587f * G + .114f * B

//Notice the trailing f's on the numbers which indicate that they are 
//single precision floating point constants and not double precision
//constants.

//You should fill in the kernel as well as set the block and grid sizes
//so that the entire image is processed.

#include "utils.h"
#include <stdio.h>



//KERNEL PARA CONVERSIÓN A HSV
__global__
void rgba_to_greyscale(const uchar4* const rgbaImage,
                       uchar4* const greyImage,
                       int numRows, int numCols)
{ 
  //TODO
  //Fill in the kernel to convert from color to greyscale
  //the mapping from components of a uchar4 to RGBA is:
  // .x -> R ; .y -> G ; .z -> B ; .w -> A
  //
  //The output (greyImage) at each pixel should be the result of
  //applying the formula: output = .299f * R + .587f * G + .114f * B;
  //Note: We will be ignoring the alpha channel for this conversion

  //First create a mapping from the 2D block and grid locations
  //to an absolute 2D location in the image, they use that to
  //calculate a 1D offset
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
  greyImage[index].x =  grey;
 greyImage[index].y =  grey;
 greyImage[index].z =  grey;
  }
}


void rgba2hsv(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
                            uchar4 * const d_greyImage, size_t numRows, size_t numCols)
{
  //You must fill in the correct sizes for the blockSize and gridSize
  //currently only one block with one thread is being launched

 // SOLUCIÓN
  int   blockWidth = 32;   // (dimensionX / gridbloqueenX) = threadsporbloqueenX
  
   //  printf("numRows = %d\n",numRows);
   // printf("numCols = %d\n",numCols);

    const dim3 blockSize(blockWidth, blockWidth, 1);
   int   blocksX = (numRows/blockWidth)+1;       // +1 por redondeo ??
   int   blocksY = numCols/blockWidth; //TODO
   const dim3 gridSize( blocksX, blocksY, 1);  //TODO

///////////////

 // const dim3 blockSize(1, 5, 1);  //TODO
 // const dim3 gridSize( 1, 10, 1);  //TODO
   rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}




