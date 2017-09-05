package com.komputation.cpu.layers.forward.encoder

import com.komputation.cpu.functions.getStep
import com.komputation.cpu.functions.setStep
import com.komputation.cpu.layers.BaseCpuForwardLayer
import com.komputation.cpu.layers.forward.units.RecurrentUnit
import com.komputation.layers.Resourceful
import com.komputation.optimization.Optimizable

class CpuSingleOutputEncoder internal constructor(
    name : String?,
    isReversed : Boolean,
    private val unit: RecurrentUnit,
    private val numberSteps: Int,
    private val inputDimension: Int,
    private val hiddenDimension : Int) : BaseCpuForwardLayer(name), Resourceful, Optimizable {

    private val startAtTheBeginning = 0..numberSteps - 1
    private val startAtTheEnd = this.numberSteps - 1 downTo 0

    private var states = emptyArray<FloatArray>()

    override val numberOutputRows = this.hiddenDimension
    override val numberOutputColumns = 1
    override var forwardResult = FloatArray(0)

    override val numberInputRows = this.inputDimension
    override val numberInputColumns = 1
    override var backwardResult = FloatArray(0)

    private val inputIndices = if(isReversed) IntArray(this.numberSteps) { index -> this.numberSteps - 1 - index } else IntArray(this.numberSteps) { index -> index }
    private var steps = emptyArray<FloatArray>()

    override fun acquire(maximumBatchSize: Int) {

        this.states = Array(this.numberSteps+1) { FloatArray(this.hiddenDimension) }
        this.steps = Array(this.numberSteps) { FloatArray(this.inputDimension) }

        this.backwardResult = FloatArray(this.inputDimension * this.numberSteps)

    }

    override fun release() {

    }

    override fun forward(withinBatch : Int, numberInputColumns : Int, input: FloatArray, isTraining : Boolean): FloatArray {

        for (indexStep in this.startAtTheBeginning) {

            val step = this.steps[indexStep]
            getStep(input, this.inputIndices[indexStep], step, this.inputDimension)

            this.states[indexStep+1] = this.unit.forwardStep(withinBatch, indexStep, this.states[indexStep], step, isTraining)

        }

        this.forwardResult = this.states[this.numberSteps]

        return this.forwardResult

    }

    override fun backward(withinBatch: Int, chain: FloatArray): FloatArray {

        var previousBackwardWrtPreviousState = chain

        for (indexStep in this.startAtTheEnd) {

            val (backwardWrtPreviousState, backwardWrtInput) = this.unit.backwardStep(withinBatch, indexStep, previousBackwardWrtPreviousState)

            previousBackwardWrtPreviousState = backwardWrtPreviousState

            setStep(backwardWrtInput, this.inputIndices[indexStep], this.backwardResult, this.inputDimension)

        }

        this.unit.backwardSeries()

        return this.backwardResult

    }

    override fun optimize(batchSize : Int) {

        if (this.unit is Optimizable) {

            this.unit.optimize(batchSize)

        }

    }

}