//Color based segmentation 

#include "utils.h"
#include <stdio.h>



//KERNEL PARA CONVERSIÓN A HSV
__global__
void rgba_2_hsv(const uchar4* const rgbaImage,
                       uchar3* const hsvImage,
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
        /// printf("index = %d\n",index);

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

}
}


__global__ void threshold_kernel(const uchar3* hsvImage,
 	     	  	    	  unsigned char* thresImage,				  
 				  int numRows, int numCols){
int Hmin=90, Smin=120, Vmin=100;
// int Hmin=100, Smin=100, Vmin=110;
//int Hmin=0, Smin=0, Vmin=0;
int Hmax=170, Smax=250, Vmax=250;

  int y = threadIdx.y+ blockIdx.y* blockDim.y;   //globalIdx = (blockIdx * threadsPerBlock) + threadId

   int x = threadIdx.x+ blockIdx.x* blockDim.x;

if (y < numCols && x < numRows) 
  {
   	int index = numRows*y +x;


unsigned char H=hsvImage[index].x;
unsigned char S=hsvImage[index].y;
unsigned char V=hsvImage[index].z;

if(H>Hmin && H<Hmax && S>Smin && S<Smax && V>Vmin && V<Vmax){
  thresImage[index]=255;  

 }else{thresImage[index]=0; }


}

 }




//KERNEL FOR EROSION

__global__ void erode_kernel(unsigned char * thresImage,
 	     	  	    	  unsigned char* erodedImage, //unsigned char* dilatedImage,
 				  int numRows, int numCols){

int x = blockIdx.x * blockDim.x +  threadIdx.x;
int y = blockIdx.y *blockDim.y + threadIdx.y;
int menor=255;

//extern  __shared__ unsigned char sh_thresImage[];

 if( y >= numCols || x>= numRows){
     return;}

    	int index = numCols*x +y;

//sh_thresImage[index]=thresImage[index];
//__syncthreads();

//Kernel with 2D VON NEUMMAN stencil pattern
//vertical values for the operator
//max and min to avoid accesing  out of bounds. a la posiciṕnen el grid se le suma una cantidad de posiciones igual al tamaño del kernel, después se desplaza por la mitad de su tamaño}

// int kernelSize = 4;
// for(int i=0; i<kernelSize;i++){
// 	int offsetX= min(max(x + i -kernelSize/2,0), numRows -1);
// 	int temp= thresImage[offsetX*numCols +y];
// 	if(temp < menor){
// 		 menor=temp;
// 		 }}

// for(int i=0; i<kernelSize;i++){    //   horizontal
// 	int offsetY= min(max(y + i -kernelSize/2,0), numCols -1);
// 	int temp= thresImage[x*numCols + offsetY];
// 	if(temp< menor){
// 		 menor=temp;
// 		 }}
 
//Kernel rectangular stencil pattern
int kernelWidth =4;
int kernelHeight =4;

for(int i=0; i<kernelWidth;i++){
 	int offsetY= min(max(y + i -kernelWidth/2,0), numCols -1);
      for(int j=0; j<kernelHeight;j++){
	int offsetX= min(max(x + i -kernelHeight/2,0), numRows -1);
	
 	int temp= thresImage[offsetX*numCols + offsetY];
	if(temp< menor){
 		 menor=temp;
 		 }

             }
	}


 erodedImage[index]=menor;
}



////KERNEL DILATACIÓN
__global__ void dilate_kernel(unsigned char * erodedImage,
 	     	  	    	  unsigned char* dilatedImage,
 				  int numRows, int numCols){

int x = blockIdx.x * blockDim.x +  threadIdx.x;
int y = blockIdx.y *blockDim.y + threadIdx.y;
int mayor=0;

if( y >= numCols || x>= numRows){
    return;}

int index = numCols*x +y;   


//Kernel with 2D VON NEUMMAN stencil pattern

//vertical values for the operator
//max and min to avoid accesing  out of bounds. a la posiciṕnen el grid se le suma una cantidad de posiciones igual al tamaño del kernel, después se desplaza por la mitad de su tamaño}

int kernelSize = 4;
for(int i=0; i<kernelSize;i++){
	int offsetX= min(max(x + i -kernelSize/2,0), numRows -1);
	int temp= erodedImage[offsetX*numCols + y];
	if(temp > mayor){
		 mayor=temp;
		 }}


for(int i=0; i<kernelSize;i++){   //horizontal values
	int offsetY= min(max(y + i -kernelSize/2,0), numCols -1);
	int temp= erodedImage[x*numCols + offsetY ];
	if(temp > mayor){
		 mayor=temp;
		 }}
		 
//Kernel rectangular stencil pattern
int kernelWidth =4;
int kernelHeight =4;

for(int i=0; i<kernelWidth;i++){
 	int offsetY= min(max(y + i -kernelWidth/2,0), numCols -1);
      for(int j=0; j<kernelHeight;j++){
	int offsetX= min(max(x + i -kernelHeight/2,0), numRows -1);
	
 	int temp= erodedImage[offsetX*numCols + offsetY];
	if(temp > mayor){
 		 mayor=temp;
 		 }}}



dilatedImage[index]=mayor;
}


__global__ void window_hgw_kernel(unsigned char * thresImage,
 	     	  	    	  unsigned char* windowImage, 
 				  int numRows, int numCols){

		int p=5;
		int apron = 2;
		int cont=0;
		int i=0;
		int indicehgw=0 ;
		int indice =0;
	
		
		for(int x=0;x<numRows;x++){
		  for(int y=0;y<=numCols;y++){  ///last bug
		    indice=x*numCols + y;//x*numRows + y ;

		    //h_erohgw_o[indice]=h_thresImage[indice];

                    if(cont==0){ //agregar apron izquierdo; inicia una nueva ventana
		      
		      for(int c=0; c<apron; c++){
		    	windowImage[indicehgw]=0;             //0:dilat|| 255||ero
		    	// printf("%u \n", h_ero_hgw[indicehgw]);
		        indicehgw++;
		        
		        }
		      
		    }

		         if (cont>=p){ //agregar apron derecho; finalizar ventana
		           
		    	   for(int c=0; c<apron; c++){
			    
		    	     windowImage[indicehgw]=0; 
		    	     // printf("%u \n", h_ero_hgw[indicehgw]);
		             indicehgw++;
			     
		    	     }
		              cont = 0; //reiniciar contador de ventana
			      y=y-1;  //regresar 1 posición en la fila
			       
		         }else{
			 
		    	    windowImage[indicehgw]=thresImage[indice];
		    	    // printf("%u  ", h_ero_hgw[indicehgw]);
		    	    // printf("%u \n", h_thresImage[indice]);
		           cont++;
		           indicehgw++;
			 
			 }
			 
			       
		    
		  }}




}


__global__ void ero_hgw_kernel(unsigned char * windowImage,
 	     	  	    	  unsigned char* erohgwImage, 
 				  int numRows, int numCols){

}





void color_seg(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
                            uchar3 * const d_hsvImage, unsigned char * d_thresImage,
			    unsigned char * d_erodedImage, unsigned char * d_dilatedImage,
			    unsigned char * d_window_hgw, unsigned char * d_ero_hgw,
			    size_t numRows, size_t numCols)
{


  
  int   blockWidth = 8;   // (dimensionX / gridbloqueenX) = threadsporbloqueenX

    const dim3 blockSize(blockWidth, blockWidth, 1);
   int   blocksX = (numRows/blockWidth)+1;       // +1 por truncamiento
   int   blocksY = numCols/blockWidth +1; 
   const dim3 gridSize( blocksX, blocksY, 1);  
   int pixel=numRows*numCols;

///////////////

  rgba_2_hsv<<<gridSize, blockSize>>>(d_rgbaImage, d_hsvImage, numRows, numCols); 
 // cudaDeviceSynchronize();
  threshold_kernel<<<gridSize, blockSize>>>(d_hsvImage, d_thresImage, numRows, numCols);
  window_hgw_kernel<<<1,1>>>(d_thresImage,d_window_hgw, numRows, numCols);
 // cudaDeviceSynchronize();

ero_hgw_kernel<<<gridSize, blockSize>>>(d_window_hgw, d_ero_hgw, numRows, numCols);

 // erode_kernel<<<gridSize,blockSize>>>(d_thresImage, d_erodedImage, numRows, numCols);
 // // cudaDeviceSynchronize();
 //  dilate_kernel<<<gridSize,blockSize>>>(d_erodedImage, d_dilatedImage,numRows, numCols);
  
  //checkCudaErrors(cudaGetLastError());
}


