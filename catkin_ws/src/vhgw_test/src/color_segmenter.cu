#include <cuda.h>
#include <iostream>

unsigned char* gpu_bgr_image;
unsigned char* gpu_hsv_image;
unsigned char* gpu_bin_image;
unsigned char* gpu_vhgw_s_array;
unsigned char* gpu_vhgw_r_array;
unsigned char* gpu_fil_image;

__global__ void bgr2hsv(unsigned char* bgr_image, unsigned char* hsv_image)
{
    int idx = blockIdx.x*blockDim.x*blockDim.y + (threadIdx.y*blockDim.x + threadIdx.x);
    idx *= 3;
    int b = bgr_image[idx];
    int g = bgr_image[idx + 1];
    int r = bgr_image[idx + 2];
    int M = max(max(b,g),r);
    int m = min(min(b,g),r);
    float C = M - m;

    if(C == 0)
    {
        hsv_image[idx    ] = 0;
        hsv_image[idx + 1] = 0;
        hsv_image[idx + 2] = 0;
        return;
    }
    int s = (int)(255*C/M);
    if(s == 0)
    {
        hsv_image[idx    ] = 0;
        hsv_image[idx + 1] = 0;
        hsv_image[idx + 2] = M;
        return;
    }

    int h=0;
    if(M == r)
    {
        h = (int)(30*(g-b)/C);
        if(h < 0) h += 180;
    }
    else if(M == g)
        h = 60 + (int)(30*(b-r)/C);
    else if(M == b)
        h = 120 + (int)(30*(b-r)/C);

    hsv_image[idx    ] = h;
    hsv_image[idx + 1] = s;
    hsv_image[idx + 2] = M;
}

__global__ void in_range(unsigned char* image_src, unsigned char* image_dst, int min_0, int min_1, int min_2, int max_0, int max_1, int max_2)
{
    int idx = blockIdx.x*blockDim.x*blockDim.y + (threadIdx.y*blockDim.x + threadIdx.x);

    unsigned char ch0 = image_src[3*idx];    
    unsigned char ch1 = image_src[3*idx + 1];
    unsigned char ch2 = image_src[3*idx + 2];
    if(ch0 > min_0 && ch0 < max_0 && ch1 > min_1 && ch1 < max_1 && ch2 > min_2 && ch2 < max_2)
        image_dst[idx] = 255;
    else
        image_dst[idx] = 0;
}

void gpu_segment_by_color(unsigned char* bgr_image, unsigned char* segmented_image, int img_rows, int img_cols, int img_channels)
{
    dim3 block_size(32,32,1);
    dim3 grid_size(300, 1,1);
    
    if(cudaMemcpy(gpu_bgr_image, bgr_image, img_rows*img_cols*img_channels, cudaMemcpyHostToDevice) != cudaSuccess)
        std::cout << "Cannot copy from host to device." << std::endl;

    bgr2hsv<<<grid_size, block_size>>>(gpu_bgr_image, gpu_hsv_image);
    if(cudaThreadSynchronize() != cudaSuccess)
        std::cout << "There were errors while launching threads." << std::endl;
    in_range<<<grid_size, block_size>>>(gpu_hsv_image, gpu_bin_image, 150,150,150,180,255,255);
    if(cudaThreadSynchronize() != cudaSuccess)
        std::cout << "There were errors while launching threads." << std::endl;
    
    cudaMemcpy(segmented_image, gpu_bin_image, img_rows*img_cols, cudaMemcpyDeviceToHost);
}

void gpu_allocate_memory(int img_rows, int img_cols, int img_channels)
{
    if(cudaSetDevice(0) != cudaSuccess) std::cout << "Cannot set GPU 0 " << std::endl;
    std::cout << "Allocating memory in GPU..." << std::endl;
    if(cudaMalloc(&gpu_bgr_image   , img_rows*img_cols*img_channels) != cudaSuccess) std::cout << "Cannot allocate" << std::endl;
    if(cudaMalloc(&gpu_hsv_image   , img_rows*img_cols*img_channels) != cudaSuccess) std::cout << "Cannot allocate" << std::endl;
    if(cudaMalloc(&gpu_bin_image   , img_rows*img_cols*img_channels) != cudaSuccess) std::cout << "Cannot allocate" << std::endl;
    if(cudaMalloc(&gpu_vhgw_s_array, img_rows*img_cols*img_channels) != cudaSuccess) std::cout << "Cannot allocate" << std::endl;
    if(cudaMalloc(&gpu_vhgw_r_array, img_rows*img_cols*img_channels) != cudaSuccess) std::cout << "Cannot allocate" << std::endl;
    if(cudaMalloc(&gpu_fil_image   , img_rows*img_cols*img_channels) != cudaSuccess) std::cout << "Cannot allocate" << std::endl;
}

void gpu_free_memory()
{
    std::cout << "Releasing memory in GPU..." << std::endl;
    if(cudaFree(gpu_bgr_image)   != cudaSuccess) std::cout << "Cannot release memory" << std::endl;
    if(cudaFree(gpu_hsv_image)   != cudaSuccess) std::cout << "Cannot release memory" << std::endl;
    if(cudaFree(gpu_bin_image)   != cudaSuccess) std::cout << "Cannot release memory" << std::endl;
    if(cudaFree(gpu_vhgw_s_array)!= cudaSuccess) std::cout << "Cannot release memory" << std::endl;
    if(cudaFree(gpu_vhgw_r_array)!= cudaSuccess) std::cout << "Cannot release memory" << std::endl;
    if(cudaFree(gpu_fil_image)   != cudaSuccess) std::cout << "Cannot release memory" << std::endl;
}
