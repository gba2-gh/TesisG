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



//EROSIÓNN///////////////////////////////////////////////////////////////////
__global__ void ero_hgw_kernel_hor(unsigned char * thresImage,
 	     	  	    	  unsigned char* suffix, 
				  unsigned char * prefix,
 				  int rows, int cols){


int offset =0;
int p=7;
unsigned char w[10]={0} ;
int c=0;
int x = blockIdx.x * blockDim.x +  threadIdx.x;
int y = blockIdx.y *blockDim.y + threadIdx.y;

//printf("y= %i\t", y);
if( y >= cols/p || x>= rows){
    return;}

int index = cols*x +y; 

//HORIZONTAL


 //arreglo s
 
   offset= p*y  + x*cols;
   
   suffix[offset + p -1] = thresImage[offset +p -1]; // agregar primer dato
   //w[c]=thresImage[offset +p -1];
   //printf("s=%u ",h_erohgw[offset+p-1]);
     for(int j=offset + p -2; j>=offset; j--){
       suffix[j]=min(thresImage[j], suffix[j+1]);
       //c++;
       //w[c]= thresImage[j];
         //printf("s=%u ",suffix[j]);
     }

     prefix[offset] = thresImage[offset] ;// agregar primer dato
     //printf("r=%u ",h_dilhgw[offset]);
     for(int j=offset + 1; j<=(offset + (p-1)); j++){
     
       prefix[j]=min(thresImage[j], prefix[j-1]);
      // c++;
       //w[c]=thresImage[j];
       // printf("r=%u ",prefix[j]);
       
    }
    
   // printf("\n");
   // printf("y= %i\t", y);
     // for(int i=0; i<= 2*(p-1); i++){
     //   printf("w=%u ",w[i]); 
     // }

     //c=0;

 
}


__global__ void ero_hgw_kernel_ver(unsigned char * thresImage,
 	     	  	    	  unsigned char* suffix, 
				  unsigned char * prefix,
 				  int rows, int cols){


int offset =0;
int p=7;
int x = blockIdx.x * blockDim.x +  threadIdx.x;
int y = blockIdx.y *blockDim.y + threadIdx.y;
 if( x>= cols || y>=rows/p){
      return;}

//printf("x=%i\t",x);
//   //VERTICAL
 //arreglo s
 
   offset=(p*y)*cols + x; 
   suffix[offset + (p -1)*cols] = thresImage[offset + (p -1)*cols]; // agregar primer dato
   
   for(int j=offset + (p -2)*cols; j>=offset; j=j-cols){
       suffix[j]=min(thresImage[j], suffix[j+cols]);
     }

   prefix[offset] = thresImage[offset] ;// agregar primer dato
   for(int j=offset + cols; j<=(offset + (p-1)*cols); j=j+cols){
     
       prefix[j]=min(thresImage[j], prefix[j-cols]);
       
    }

 
}
/////////////////////////////////////////////////////EROSIÓN/////////////////////////////////////



/////DILATACIÓN///////////////////////////////////////////////////////////////////////////////


__global__ void dil_hgw_kernel_hor(unsigned char * erohgw,
 	     	  	    	  unsigned char* suffix, 
				  unsigned char * prefix,
 				  int rows, int cols){


int offset =0;
int p=7;

int x = blockIdx.x * blockDim.x +  threadIdx.x;
int y = blockIdx.y *blockDim.y + threadIdx.y;

if( y >= cols/p || x>= rows){
    return;}

///////HORIZONTAL////////

 //arreglo s
 
   offset= p*y  + x*cols;
   
   suffix[offset + p -1] = erohgw[offset +p -1]; // agregar primer dato
     for(int j=offset + p -2; j>=offset; j--){
       suffix[j]=max(erohgw[j], suffix[j+1]);
     }

     prefix[offset] = erohgw[offset] ;// agregar primer dato

   for(int j=offset + 1; j<=(offset + (p-1)); j++){    
       prefix[j]=max(erohgw[j], prefix[j-1]);   
    }


 
}


__global__ void dil_hgw_kernel_ver(unsigned char * thresImage,
 	     	  	    	  unsigned char* suffix, 
				  unsigned char * prefix,
 				  int rows, int cols){


int offset =0;
int p=7;
int x = blockIdx.x * blockDim.x +  threadIdx.x;
int y = blockIdx.y *blockDim.y + threadIdx.y;
 if( x>= cols || y>=rows/p){
      return;}

//printf("x=%i\t",x);
//   //VERTICAL
 //arreglo s
   offset=(p*y)*cols + x; 
   suffix[offset + (p -1)*cols] = thresImage[offset + (p -1)*cols]; // agregar primer dato
   
   for(int j=offset + (p -2)*cols; j>=offset; j=j-cols){
       suffix[j]=max(thresImage[j], suffix[j+cols]);
     }

   prefix[offset] = thresImage[offset] ;// agregar primer dato
   for(int j=offset + cols; j<=(offset + (p-1)*cols); j=j+cols){
     
       prefix[j]=max(thresImage[j], prefix[j-cols]);
       
    }


}

///////////////////////////////////////////////////////DILATACIÓN//////////////////////////


__global__ void ero_result_kernel(unsigned char *suffix, unsigned char *prefix, unsigned char *  erohgw, int rows, int cols){

int x = blockIdx.x * blockDim.x +  threadIdx.x;
int y = blockIdx.y *blockDim.y + threadIdx.y;

if( y >= cols || x>= rows){
    return;}
    
//printf("y= %i\t", y);
int index = cols*x +y; 
     erohgw[index]=min(suffix[index],prefix[index]); //result 
}


__global__ void dil_result_kernel(unsigned char *suffix, unsigned char *prefix, unsigned char *  erohgw, int rows, int cols){

int x = blockIdx.x * blockDim.x +  threadIdx.x;
int y = blockIdx.y *blockDim.y + threadIdx.y;

if( y >= cols || x>= rows){
    return;}

int index = cols*x +y; 
     erohgw[index]=max(suffix[index],prefix[index]); //result 
}




void color_seg(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
                            uchar3 * const d_hsvImage, unsigned char * d_thresImage,
			    unsigned char * d_erodedImage, unsigned char * d_dilatedImage,
			    unsigned char * d_erohgw,unsigned char * d_dilhgw, unsigned char * d_suffix, unsigned char *d_prefix,
			    size_t numRows, size_t numCols)
{


  int p=5;
  int   blockWidth = 8;   // (dimensionX / gridbloqueenX) = threadsporbloqueenX

   const dim3 blockSize(blockWidth, blockWidth, 1);
   int   blocksX = (numRows/blockWidth)+1;       // +1 por truncamiento
   int   blocksY = (numCols/blockWidth) +1 ; 
   const dim3 gridSize( blocksX, blocksY , 1);  
  

///////////////

  rgba_2_hsv<<<gridSize, blockSize>>>(d_rgbaImage, d_hsvImage, numRows, numCols); 
 // cudaDeviceSynchronize();
  threshold_kernel<<<gridSize, blockSize>>>(d_hsvImage, d_thresImage, numRows, numCols);

 // cudaDeviceSynchronize();

ero_hgw_kernel_hor<<<gridSize,blockSize,1>>>(d_thresImage, d_suffix, d_prefix,  numRows, numCols);
ero_result_kernel<<<gridSize, blockSize,1>>>(d_suffix, d_prefix, d_erohgw, numRows, numCols);
ero_hgw_kernel_ver<<<gridSize,blockSize,1>>>(d_erohgw, d_suffix, d_prefix,  numRows, numCols);
ero_result_kernel<<<gridSize, blockSize,1>>>(d_suffix, d_prefix, d_erohgw, numRows, numCols);

cudaDeviceSynchronize();

dil_hgw_kernel_hor<<<gridSize, blockSize,1>>>(d_erohgw, d_suffix, d_prefix,  numRows, numCols);
dil_result_kernel<<<gridSize, blockSize,1>>>(d_suffix, d_prefix, d_dilhgw, numRows, numCols);
dil_hgw_kernel_ver<<<gridSize, blockSize,1>>>(d_dilhgw, d_suffix, d_prefix,  numRows, numCols);
dil_result_kernel<<<gridSize, blockSize,1>>>(d_suffix, d_prefix, d_dilhgw, numRows, numCols);






 // erode_kernel<<<gridSize,blockSize>>>(d_thresImage, d_erodedImage, numRows, numCols);
 // // cudaDeviceSynchronize();
 //  dilate_kernel<<<gridSize,blockSize>>>(d_erodedImage, d_dilatedImage,numRows, numCols);
  
  //checkCudaErrors(cudaGetLastError());
}


