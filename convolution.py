import numpy as np
from numba import jit, cuda
import cv2
import time

@cuda.jit
def cuda_convolve(matrix, kernel, output, pad):
    y, x = cuda.grid(2)
    result = 0
    if(y >= matrix.shape[0] or x >= matrix.shape[1]):
        return

    for i in range(0, kernel.shape[0]):
        for j in range(0, kernel.shape[1]):
            result += matrix[y + i, x + j] * kernel[i, j]
    output[y, x] = result

def convolve(matrix, kernel):
    #find padding for matrix based on kernel width
    pad = int((kernel.shape[0] - (kernel.shape[0] % 2))/2)
    #add padding to border of matrix
    padded_matrix = np.zeros((matrix.shape[0] +  2 * pad, matrix.shape[1] + 2 * pad))
    padded_matrix[pad : -pad, pad: -pad] = matrix
    #threads per block
    TPB = (32, 32)
    #blocks per grid
    bpg_x = int(np.ceil(matrix.shape[0]/TPB[0]))
    bpg_y = int(np.ceil(matrix.shape[1]/TPB[1]))
    BPG = (bpg_x, bpg_y)
    #define output
    output = np.empty_like(matrix)
    cuda_convolve[BPG, TPB](padded_matrix, kernel, output, pad)
    return output

@jit(nopython=True)
def convolve2d(image, kernel):
    pad = int((kernel.shape[0] - (kernel.shape[1] % 2))/2)
    padded_image = np.zeros((image.shape[0] + 2 * pad, image.shape[1] + 2 * pad))
    padded_image[pad : - pad, pad : -pad] = image
    product = np.empty_like(padded_image)
   
    for i in range(pad, pad + image.shape[0]):
        for j in range(pad, pad + image.shape[1]):
            product[i, j] = np.sum(padded_image[i - pad : i + pad + 1, j - pad : j + pad + 1] * kernel)
    return product[pad : -pad, pad : -pad]

def main():
    #matrix height, width and kernel size
    x_lim = 1024
    y_lim = 1024
    k_width = 7
    
    #generate random matrix and kernel
    kernel = np.random.rand(k_width, k_width)
    matrix = np.random.rand(y_lim, x_lim)
    
    #perform convolutions
    start = time.time()
    output = convolve(matrix, kernel)
    t1 = time.time()
    print("GPU time", t1 - start)
    start = time.time()
    output = convolve2d(matrix, kernel)
    t1 = time.time()
    print("CPU time", t1 - start)

if __name__ == "__main__":
    main()