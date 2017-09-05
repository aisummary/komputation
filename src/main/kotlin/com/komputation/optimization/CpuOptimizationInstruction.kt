package com.komputation.optimization

import com.komputation.cpu.optimization.CpuOptimizationStrategy

interface CpuOptimizationInstruction {

    fun buildForCpu() : CpuOptimizationStrategy

}