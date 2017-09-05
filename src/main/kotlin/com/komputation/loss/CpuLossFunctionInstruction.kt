package com.komputation.loss

import com.komputation.cpu.loss.CpuLossFunction

interface CpuLossFunctionInstruction {

    fun buildForCpu() : CpuLossFunction

}