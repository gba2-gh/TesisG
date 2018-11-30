#include <string>
#include <iostream>
#include <stdio.h>

int main (){

float rgbaMAX=0;
float rgbaMIN=0;

unsigned char Rp= 200, Gp= 100, Bp=50;

//CONVERT 8 B TO FLOAT 
float B=Bp*(1.0/255.0), G=Gp*(1.0/255.0), R=Rp*(1.0/255.0);

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
     
Hp=(60*(G-B))/(rgbaMAX-rgbaMIN) +360;
   }
   }
 if(V==G*255){ Hp=(120+60*(B-R))/(rgbaMAX-rgbaMIN);}
if(V==B*255){ Hp=(240+60*(R-G))/(rgbaMAX-rgbaMIN);}
H=Hp*(0.5);
 
// printf("Hp= %f", Hp);
 printf("H= %u", H);
  printf("S= %u", S);
   printf("V= %u", V);

   
   return 0;
 } 
