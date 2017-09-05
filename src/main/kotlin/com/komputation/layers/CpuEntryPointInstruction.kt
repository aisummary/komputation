package com.komputation.layers

import com.komputation.cpu.layers.CpuEntryPoint

interface CpuEntryPointInstruction {

    fun buildForCpu() : CpuEntryPoint

}