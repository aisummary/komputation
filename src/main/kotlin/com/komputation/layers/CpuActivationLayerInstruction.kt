package com.komputation.layers

import com.komputation.cpu.layers.forward.activation.CpuActivationLayer

interface CpuActivationLayerInstruction : CpuForwardLayerInstruction {

    override fun buildForCpu(): CpuActivationLayer

}