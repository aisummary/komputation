package com.komputation.layers.forward

import com.komputation.cpu.layers.forward.CpuCounterProbabilityLayer
import com.komputation.layers.CpuForwardLayerInstruction

class CounterProbabilityLayer internal constructor(
    private val name : String?,
    private val numberRows: Int,
    private val numberColumns: Int) : CpuForwardLayerInstruction {

    override fun buildForCpu() =

        CpuCounterProbabilityLayer(this.name, this.numberRows, this.numberColumns)

}


fun counterProbabilityLayer(numberRows: Int, numberColumns: Int) =

    counterProbabilityLayer(null, numberRows, numberColumns)

fun counterProbabilityLayer(name : String?, numberRows: Int, numberColumns: Int) =

    CounterProbabilityLayer(name, numberRows, numberColumns)