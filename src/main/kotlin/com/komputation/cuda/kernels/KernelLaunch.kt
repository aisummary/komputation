package com.komputation.cuda.kernels

import jcuda.Pointer
import jcuda.driver.CUfunction
import jcuda.driver.JCudaDriver.cuLaunchKernel

fun launchKernel(function : CUfunction, parameters : Pointer, numberBlocksInXDimension: Int, numberBlocksInYDimension: Int, numberThreadsPerBlock: Int, sharedMemoryBytes : Int) =

    cuLaunchKernel(
        function,
        numberBlocksInXDimension, numberBlocksInYDimension, 1,
        numberThreadsPerBlock, 1, 1,
        sharedMemoryBytes,
        null,
        parameters,
        null
    )