package com.komputation.cuda.kernels.launch

import com.komputation.matrix.IntMath

fun computeNumberOfThreadsForRows(numberRows : Int, warpSize: Int, maximumNumberThreadsPerBlock : Int) : Pair<Int, Int> {

    if (numberRows < maximumNumberThreadsPerBlock) {

        val numberIterations = 1

        val numberWarps = (numberRows + warpSize - 1) / warpSize
        val numberThreads = numberWarps * warpSize

        return numberIterations to numberThreads

    }
    else {

        val numberIterations = (numberRows + maximumNumberThreadsPerBlock - 1) / maximumNumberThreadsPerBlock

        val iterativeWarpSize = warpSize * numberIterations
        val numberIterativeWarps = (numberRows + iterativeWarpSize - 1) / iterativeWarpSize
        val numberThreads = numberIterativeWarps * warpSize

        return numberIterations to numberThreads

    }

}

fun computeEntrywiseLaunchConfiguration(
    numberElements : Int,
    numberMultiProcessors : Int,
    numberResidentWarps : Int,
    warpSize : Int,
    maximumNumberThreadsPerBlock : Int): KernelLaunchConfiguration {

    val numberRequiredWarps = (numberElements + warpSize - 1) / warpSize

    // No more than one block per multi-processor
    if (numberRequiredWarps <= numberMultiProcessors) {

        val numberBlocks = (numberElements + warpSize - 1) / warpSize

        return KernelLaunchConfiguration(numberBlocks, warpSize, 1)

    }
    // More than one block per multi-processor
    else {

        val numberAvailableWarps = numberResidentWarps * numberMultiProcessors

        val numberOccupiedWarps = IntMath.min(numberRequiredWarps, numberAvailableWarps)
        val numberOccupiedWarpsPerMultiprocessor = (numberOccupiedWarps + numberMultiProcessors - 1) / numberMultiProcessors

        val numberRequiredThreadsPerMultiprocessor = numberOccupiedWarpsPerMultiprocessor * warpSize
        val numberThreadsPerBlock = Math.min(numberRequiredThreadsPerMultiprocessor, maximumNumberThreadsPerBlock)

        val numberBlocksPerMultiprocessor = (numberRequiredThreadsPerMultiprocessor + maximumNumberThreadsPerBlock - 1) / maximumNumberThreadsPerBlock

        val numberBlocks = numberMultiProcessors * numberBlocksPerMultiprocessor

        val numberOccupiedThreads = numberBlocks * maximumNumberThreadsPerBlock

        val numberIterations = (numberElements + numberOccupiedThreads - 1) / numberOccupiedThreads

        return KernelLaunchConfiguration(numberBlocks, numberThreadsPerBlock, numberIterations)

    }

}