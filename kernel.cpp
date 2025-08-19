
#include "opengl.h"

__global__ void pixelKernel(uchar4* pixels, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    float u = ((x / (float)width) * 2.0f - 1.0f) * (width / (float)height);
    float v = (y / (float)height) * 2.0f - 1.0f;

    int index = -1;
    if ((x < width) && (y < height))
        index = y * width + x;
    else
        return;

    if ((u * u) + (v * v) < 1.0f)
        pixels[index] = make_uchar4(255, 255, 255, 255);
    else
        pixels[index] = make_uchar4(0, 0, 0, 255);
}

void InteropOpenGL::executePixelKernel()
{
    uchar4* pixels = nullptr;
    size_t size = 0;
    cudaGraphicsMapResources(1, &cudaPBO, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&pixels, &size, cudaPBO);
    pixelKernel <<<grid, block>>> (pixels, screenWidth, screenHeight);
    cudaDeviceSynchronize();
}

