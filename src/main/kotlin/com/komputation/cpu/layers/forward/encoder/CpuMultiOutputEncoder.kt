package com.komputation.cpu.layers.forward.encoder

import com.komputation.cpu.functions.add
import com.komputation.cpu.functions.getStep
import com.komputation.cpu.functions.setStep
import com.komputation.cpu.layers.BaseCpuForwardLayer
import com.komputation.cpu.layers.forward.units.RecurrentUnit
import com.komputation.layers.Resourceful
import com.komputation.optimization.Optimizable

class CpuMultiOutputEncoder internal constructor(
    name : String?,
    isReversed : Boolean,
    private val unit: RecurrentUnit,
    private val numberSteps: Int,
    private val inputDimension: Int,
    private val hiddenDimension : Int) : BaseCpuForwardLayer(name), Resourceful, Optimizable {

    private val startAtTheBeginning = 0..numberSteps - 1
    private val startAtTheEnd = this.numberSteps - 1 downTo 0

    private var inputIndices = if(isReversed) IntArray(this.numberSteps) { index -> this.numberSteps - 1 - index } else IntArray(this.numberSteps) { index -> index }
    private var steps = emptyArray<FloatArray>()
    private var states = emptyArray<FloatArray>()
    private var previousBackwardStatePreActivationWrtPreviousState = FloatArray(0)

    override val numberOutputRows = this.hiddenDimension
    override val numberOutputColumns = this.numberSteps
    override var forwardResult = FloatArray(0)

    override val numberInputRows = this.inputDimension
    override val numberInputColumns = this.numberSteps
    override var backwardResult = FloatArray(0)

    override fun acquire(maximumBatchSize: Int) {

        this.steps = Array(this.numberSteps) { FloatArray(this.inputDimension) }
        this.states = Array(this.numberSteps+1) { FloatArray(this.hiddenDimension) }

        this.forwardResult = FloatArray(this.numberOutputColumns * this.numberOutputRows)

        this.backwardResult = FloatArray(this.numberInputColumns * numberInputRows)

    }

    override fun release() {

    }

    override fun forward(withinBatch : Int, numberInputColumns : Int, input: FloatArray, isTraining : Boolean): FloatArray {

        for (indexStep in this.startAtTheBeginning) {

            val step = this.steps[indexStep]
            getStep(input, this.inputIndices[indexStep], step, this.inputDimension)

            val newState = this.unit.forwardStep(withinBatch, indexStep, this.states[indexStep], step, isTraining)

            this.states[indexStep+1] = newState

            System.arraycopy(newState, 0, this.forwardResult, indexStep * this.hiddenDimension, this.hiddenDimension)

        }

        return this.forwardResult

    }

    private val chainStep = FloatArray(this.hiddenDimension)
    private var backwardSumWrtState = FloatArray(this.hiddenDimension)

    override fun backward(withinBatch : Int, chain: FloatArray): FloatArray {

        for (indexStep in this.startAtTheEnd) {

            getStep(chain, indexStep, this.chainStep, this.hiddenDimension)

            if (indexStep + 1 == this.numberSteps) {

                this.backwardSumWrtState = this.chainStep

            }
            else {

                add(this.previousBackwardStatePreActivationWrtPreviousState, this.chainStep, this.backwardSumWrtState, this.hiddenDimension)

            }

            val (backwardStatePreActivationWrtPreviousState, backwardStatePreActivationWrtInput) = this.unit.backwardStep(withinBatch, indexStep, this.backwardSumWrtState)

            this.previousBackwardStatePreActivationWrtPreviousState = backwardStatePreActivationWrtPreviousState

            setStep(backwardStatePreActivationWrtInput, indexStep, this.backwardResult, this.inputDimension)

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