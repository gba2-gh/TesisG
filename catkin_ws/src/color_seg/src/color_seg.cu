//Color based segmentation 

#include "utils.h"
#include <stdio.h>



//KERNEL PARA CONVERSIÓN A HSV
__global__
void rgba_2_hsv(const uchar4* const rgbaImage,
                       uchar4* const hsvImage,
                       int numRows, int numCols)
{ 

  int y = threadIdx.y+ blockIdx.y* blockDim.y;   //globalIdx = (blockIdx * threadsPerBlock) + threadId

  int x = threadIdx.x+ blockIdx.x* blockDim.x;
float rgbaMAX=0;
float rgbaMIN=0;

//prevents accessing out of bounds
if (y < numCols && x < numRows) 
  {
  	int index = numRows*y +x;    ///numCols
        /// printf("index = %d\n",index);}

//CONVERT 8 B TO FLOAT 
float R=rgbaImage[index].x*(1.0/255.0), G=rgbaImage[index].y*(1.0/255.0), B=rgbaImage[index].z*(1.0/255.0);

//FIND MAX AND MIN VALUES FOR THE RGB STRUCT

if(B > G){
	if(B > R){
	     rgbaMAX= B; //B CHANNEL MAX VAlUE
	    
	       if(G > R){
	        	rgbaMIN= R;}
	       else{rgbaMIN= G;}
	}else{rgbaMAX=R;
		rgbaMIN=G;}
  }else{
	if(G > R){
	      rgbaMAX= G;
	      if(B > R){
	      rgbaMIN= R;}
	      else{rgbaMIN= B;}
	}else{rgbaMAX= R;
	      rgbaMIN= B;}
  }


// printf("rgbaMAX= %f\n",rgbaMAX);
// printf("rgbaMIN= %f\n",rgbaMIN);

unsigned char V = rgbaMAX*(255); /// V=MAX(R,G,B)
unsigned char S=0;
unsigned char H=0;
float Sp=0, Hp=0;

//Saturation
if(V != 0)
  {Sp=((rgbaMAX-rgbaMIN)/rgbaMAX); } ///  S= (V-min(R,G,B)) / V }
S=Sp*(255);

//hue ineficiente
if(V==R*255){
   if(G>=B){
     Hp=(60*(G-B))/(rgbaMAX-rgbaMIN);}
 else{    
	Hp=((60*(G-B))/(rgbaMAX-rgbaMIN) )+360;}
}
if(V==G*255){Hp=((60*(B-R))/(rgbaMAX-rgbaMIN))+120;}
if(V==B*255 && V!=G*255 && V!=R*255){ Hp=((60*(R-G))/(rgbaMAX-rgbaMIN))+240;}
H=Hp*(0.5);

if(H==0){H=1;}
 

hsvImage[index].x= H;
hsvImage[index].y= S;
hsvImage[index].z= V;
//printf("V=%u", V);
//printf("S=%u", S);
//printf("%u\n",H);


//     //grey consersion
//   uchar4 color = rgbaImage[index];
// // printf("color.x = %u\n",(unsigned char)(color.x)); 
//   unsigned char grey =  (unsigned char)(0.299f*color.x+ 0.587f*color.y + 0.114f*color.z);
//   greyImage[index].x = grey ;
//  greyImage[index].y =  grey;
//  greyImage[index].z =  grey;
  
}
}


void rgba2hsv(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
                            uchar4 * const d_greyImage, size_t numRows, size_t numCols)
{
  
  int   blockWidth = 32;   // (dimensionX / gridbloqueenX) = threadsporbloqueenX

    const dim3 blockSize(blockWidth, blockWidth, 1);
   int   blocksX = (numRows/blockWidth)+1;       // +1 por redondeo ??
   int   blocksY = numCols/blockWidth +1; //TODO
   const dim3 gridSize( blocksX, blocksY, 1);  //TODO

///////////////

   rgba_2_hsv<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
  cudaDeviceSynchronize(); 
  checkCudaErrors(cudaGetLastError());
}




