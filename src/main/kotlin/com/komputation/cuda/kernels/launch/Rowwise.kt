package com.komputation.cuda.kernels.launch

fun computeRowwiseLaunchConfiguration(
    numberRows : Int,
    numberColumns : Int,
    maximumNumberThreadsPerBlock : Int) : KernelLaunchConfiguration {

    if (numberColumns <= maximumNumberThreadsPerBlock) {

        return KernelLaunchConfiguration(numberRows, numberColumns, 1)

    }
    else {

        val numberIterations = (numberColumns + maximumNumberThreadsPerBlock - 1) / maximumNumberThreadsPerBlock

        return KernelLaunchConfiguration(numberRows, maximumNumberThreadsPerBlock, numberIterations)

    }

}
