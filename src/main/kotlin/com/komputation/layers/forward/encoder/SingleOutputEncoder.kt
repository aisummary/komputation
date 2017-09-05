package com.komputation.layers.forward.encoder

import com.komputation.cpu.layers.forward.encoder.CpuSingleOutputEncoder
import com.komputation.cpu.layers.forward.units.RecurrentUnit
import com.komputation.layers.CpuForwardLayerInstruction

class SingleOutputEncoder internal constructor(
    private val name : String?,
    private val unit : RecurrentUnit,
    private val numberSteps : Int,
    private val inputDimension : Int,
    private val hiddenDimension: Int,
    private val isReversed: Boolean) : CpuForwardLayerInstruction {

    override fun buildForCpu() =

        CpuSingleOutputEncoder(
            this.name,
            this.isReversed,
            this.unit,
            this.numberSteps,
            this.inputDimension,
            this.hiddenDimension
        )


}

fun singleOutputEncoder(
    unit : RecurrentUnit,
    numberSteps : Int,
    inputDimension : Int,
    hiddenDimension: Int,
    isReversed: Boolean = false) =

    singleOutputEncoder(
        null,
        unit,
        numberSteps,
        inputDimension,
        hiddenDimension,
        isReversed
    )

fun singleOutputEncoder(
    name : String?,
    unit : RecurrentUnit,
    numberSteps : Int,
    inputDimension : Int,
    hiddenDimension: Int,
    isReversed: Boolean = false) =

    SingleOutputEncoder(name, unit, numberSteps, inputDimension, hiddenDimension, isReversed)