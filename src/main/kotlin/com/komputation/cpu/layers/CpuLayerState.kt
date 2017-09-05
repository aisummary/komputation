package com.komputation.cpu.layers

interface CpuForwardState {

    val forwardResult: FloatArray
    val numberOutputRows: Int
    val numberOutputColumns: Int

}

interface CpuBackwardState {

    val backwardResult: FloatArray
    val numberInputRows: Int
    val numberInputColumns: Int

}


interface CpuLayerState : CpuForwardState, CpuBackwardState