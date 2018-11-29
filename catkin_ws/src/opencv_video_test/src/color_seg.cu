
#include <stdio.h>

//revisar tipos de datos de uchar4 * y const uchar4
__global__
void rgba_to_hsv(const uchar4 * const rgbaImage, uchar4 * const hsvImage, size_t rows, size_t cols ){

//encontrar posición global en el grid

int idxX= blockIdx.x * blockDim.x + threadIdx.x;
int idxY= blockIdx.y* blockDim.y + threadIdx.y;

// índice global
int index= idxY*cols+idxX;

//printf("index= %d \n",index);
//printf("idxX= %d \n",idxX);


//evita acceder a memoria fuera de la imagen
if (idxX > cols || idxY> rows){
printf("si");
return;}

printf("no \n");
int rgbaMAX=0;
int rgbaMIN=0;


//ILLEGAL MEMORY ACCESS
///
hsvImage[1]=rgbaImage[1];
//printf("%u \n",color.x);

//encontrar máximo y mínimo de la structura RGB
// if(rgbaImage[index].x > rgbaImage[index].y){
//printf("1    \n");
// 	if(rgbaImage[index].x > rgbaImage[index].z){
// 	     rgbaMAX= rgbaImage[index].x;
// 	       if(rgbaImage[index].y > rgbaImage[index].z){
// 	        	rgbaMIN= rgbaImage[index].z;}
// 	       else{rgbaMIN= rgbaImage[index].y;}
// 	}else{rgbaMAX=rgbaImage[index].z;
// 		rgbaMIN=rgbaImage[index].y;}
//   }else{
//printf("0    \n");
// 	if(rgbaImage[index].y > rgbaImage[index].z){
// 	      rgbaMAX= rgbaImage[index].y;
// 	      if(rgbaImage[index].x > rgbaImage[index].z){
// 	      rgbaMIN= rgbaImage[index].z;}
// 	      else{rgbaMIN= rgbaImage[index].x;}
// 	}else{rgbaMAX= rgbaImage[index].z;
// 	      rgbaMIN= rgbaImage[index].x;}
//   }


//printf("hola \n");
//hsvImage[index].x= rgbaMAX; //h

//printf("%zu \n",hsvImage[index].x);
//printf("HSV.x= %d \n",hsvImage[index].x);
//hsvImage[index].y= //s
//hsvImage[index].z= (rgbaImage[index].x,rgbaImage[index].y,rgbaImage[index].z)        //v

}

void prueba_cuda(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage, uchar4 * const d_hsvImage,size_t rows, size_t cols){

//tamaño del bloque y la malla de threads
const dim3 blocksize(32,32,1);
const dim3 gridsize(cols/blocksize.x + 1,rows/blocksize.y + 1,1);

//printf("hola?");

//el kernel tiene como entrada la imagen rgba y como salida la imagen en hsv
rgba_to_hsv<<<gridsize,blocksize>>>(d_rgbaImage, d_hsvImage, rows, cols);

cudaDeviceSynchronize();



}
