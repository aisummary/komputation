package com.komputation.cuda.layers.forward.activation

import com.komputation.cuda.kernels.Kernel

class CudaTanhLayer internal constructor(
    name : String? = null,
    numberRows : Int,
    numberColumns : Int,
    createForwardKernel: () -> Kernel,
    createBackwardKernel: () -> Kernel,
    warpSize : Int,
    maximumNumberThreadsPerBlock : Int) : BaseCudaEntrywiseActivationLayer(name, createForwardKernel, createBackwardKernel, numberRows, numberColumns, warpSize, maximumNumberThreadsPerBlock)