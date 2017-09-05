package com.komputation.layers

import com.komputation.cpu.layers.CpuForwardLayer

interface CpuForwardLayerInstruction {

    fun buildForCpu() : CpuForwardLayer

}