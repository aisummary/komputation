package com.komputation.cuda.kernels.launch

data class KernelLaunchConfiguration(
    val numberBlocks : Int,
    val numberThreadsPerBlock : Int,
    val numberIterations : Int)