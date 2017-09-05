package com.komputation.layers.forward

import com.komputation.cpu.layers.forward.CpuConcatenation
import com.komputation.layers.CpuForwardLayerInstruction

class Concatenation internal constructor(
    private val name : String?,
    private val numberInputRows : Int,
    private val numberInputColumns : Int,
    private val hasFixedLength : Boolean,
    private val heights: IntArray,
    private val width: Int,
    private val layerInstructions: Array<CpuForwardLayerInstruction>) : CpuForwardLayerInstruction {

    private val minimumColumns = if(this.hasFixedLength) this.numberInputColumns else 1
    private val maximumColumns = this.numberInputColumns

    override fun buildForCpu() =

        CpuConcatenation(this.name, this.numberInputRows, this.minimumColumns, this.maximumColumns, this.heights, this.width, this.layerInstructions.map { instruction -> instruction.buildForCpu() }.toTypedArray())

}

fun concatenation(numberInputRows : Int, numberInputColumns : Int, hasFixedLength: Boolean, numbersOutputRows: IntArray, numberOutputColumns: Int, layers: Array<CpuForwardLayerInstruction>) =

    concatenation(null, numberInputRows, numberInputColumns, hasFixedLength, numbersOutputRows, numberOutputColumns, layers)

fun concatenation(name : String?, numberInputRows : Int, numberInputColumns : Int, hasFixedLength: Boolean, numbersOutputRows: IntArray, numberOutputColumns: Int, layers: Array<CpuForwardLayerInstruction>) =

    Concatenation(name, numberInputRows, numberInputColumns, hasFixedLength, numbersOutputRows, numberOutputColumns, layers)