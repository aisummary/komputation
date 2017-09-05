package com.komputation.layers.forward

import com.komputation.cpu.layers.forward.CpuColumnRepetitionLayer
import com.komputation.layers.CpuForwardLayerInstruction

class ColumnRepetitionLayer internal constructor(
    private val name : String?,
    private val numberRows : Int,
    private val numberColumns : Int) : CpuForwardLayerInstruction {

    override fun buildForCpu() =

        CpuColumnRepetitionLayer(this.name, this.numberRows, this.numberColumns)

}

fun columnRepetitionLayer(numberRows : Int, numberColumns : Int) =

    ColumnRepetitionLayer(null, numberRows, numberColumns)

fun columnRepetitionLayer(name : String? = null, numberRows : Int, numberColumns : Int) =

    ColumnRepetitionLayer(name, numberRows, numberColumns)