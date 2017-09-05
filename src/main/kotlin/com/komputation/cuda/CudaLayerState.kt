package com.komputation.cuda

import jcuda.Pointer

interface CudaForwardState {

    val deviceForwardResult: Pointer
    val numberOutputRows : Int
    val maximumOutputColumns : Int

}

interface CudaBackwardState {

    val deviceBackwardResult: Pointer
    val numberInputRows : Int
    val maximumInputColumns : Int

}

interface CudaLayerState : CudaForwardState, CudaBackwardState