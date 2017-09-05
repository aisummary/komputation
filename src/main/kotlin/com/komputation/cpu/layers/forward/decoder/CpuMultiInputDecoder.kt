package com.komputation.cpu.layers.forward.decoder

import com.komputation.cpu.functions.add
import com.komputation.cpu.functions.getStep
import com.komputation.cpu.functions.setStep
import com.komputation.cpu.layers.BaseCpuForwardLayer
import com.komputation.cpu.layers.forward.activation.CpuActivationLayer
import com.komputation.cpu.layers.forward.projection.SeriesBias
import com.komputation.cpu.layers.forward.projection.SeriesWeighting
import com.komputation.cpu.layers.forward.units.RecurrentUnit
import com.komputation.layers.Resourceful
import com.komputation.optimization.Optimizable

class CpuMultiInputDecoder internal constructor(
    name : String?,
    private val numberSteps : Int,
    private val inputDimension: Int,
    private val hiddenDimension: Int,
    private val outputDimension : Int,
    private val unit : RecurrentUnit,
    private val weighting: SeriesWeighting,
    private val bias: SeriesBias?,
    private val activations: Array<CpuActivationLayer>) : BaseCpuForwardLayer(name), Resourceful, Optimizable {

    private var states = emptyArray<FloatArray>()
    private var steps = emptyArray<FloatArray>()

    override val numberOutputRows = this.outputDimension
    override val numberOutputColumns = this.numberSteps
    override var forwardResult = FloatArray(0)

    override val numberInputRows = this.inputDimension
    override val numberInputColumns = this.numberSteps
    override var backwardResult = FloatArray(0)

    override fun acquire(maximumBatchSize: Int) {

        this.states = Array(this.numberSteps+1) { FloatArray(this.hiddenDimension) }
        this.steps = Array(this.numberSteps) { FloatArray(this.inputDimension) }

        this.forwardResult = FloatArray(this.numberSteps * this.outputDimension)
        this.backwardResult = FloatArray(this.numberSteps * this.inputDimension)

    }

    override fun release() {

    }

    override fun forward(withinBatch : Int, numberInputColumns : Int, input: FloatArray, isTraining : Boolean): FloatArray {

        for (indexStep in 0..this.numberSteps - 1) {

            val step = this.steps[indexStep]
            getStep(input, indexStep, step, this.inputDimension)

            // Compute the new state
            val newState = this.unit.forwardStep(withinBatch, indexStep, this.states[indexStep], step, isTraining)
            this.states[indexStep+1] = newState

            val output = this.forwardOutput(withinBatch, indexStep, newState, isTraining)

            // Store the n-th output
            setStep(output, indexStep, this.forwardResult, this.outputDimension)

        }

        return this.forwardResult

    }

    private fun forwardOutput(withinBatch : Int, indexStep: Int, state: FloatArray, isTraining : Boolean): FloatArray {

        val weighted = this.weighting.forwardStep(withinBatch, indexStep, state, isTraining)

        val biased =

            if (this.bias != null) {

                this.bias.forwardStep(withinBatch, indexStep, weighted, isTraining)

            }
            else {

                weighting.forwardResult
            }

        return this.activations[indexStep].forward(withinBatch, 1, biased, isTraining)

    }

    private val chainStep = FloatArray(this.outputDimension)
    private val backwardPreActivationSum = FloatArray(this.hiddenDimension)

    // Incoming gradient: d chain / d series prediction
    override fun backward(withinBatch : Int, chain: FloatArray): FloatArray {

        var backwardPreActivationWrtPreviousState : FloatArray? = null

        for (indexStep in this.numberSteps - 1 downTo 0) {

            getStep(chain, indexStep, this.chainStep, this.outputDimension)

            val backwardPreActivationWrtState = this.backwardOutput(withinBatch, indexStep, this.chainStep)

            if (backwardPreActivationWrtPreviousState != null) {

                add(backwardPreActivationWrtPreviousState, backwardPreActivationWrtState, this.backwardPreActivationSum, this.hiddenDimension)

            }
            else {

                System.arraycopy(backwardPreActivationWrtState, 0, this.backwardPreActivationSum, 0, this.hiddenDimension)

            }

            val (newBackwardPreActivationWrtPreviousState, newBackwardPreActivationWrtInput) = this.unit.backwardStep(withinBatch, indexStep, this.backwardPreActivationSum)

            setStep(newBackwardPreActivationWrtInput, indexStep, this.backwardResult, this.inputDimension)
            backwardPreActivationWrtPreviousState = newBackwardPreActivationWrtPreviousState

        }

        this.unit.backwardSeries()

        this.weighting.backwardSeries()
        this.bias?.backwardSeries()

        return this.backwardResult

    }

    private fun backwardOutput(withinBatch : Int, indexStep: Int, chainStep: FloatArray): FloatArray {

        val activation = this.activations[indexStep]
        activation.backward(withinBatch, chainStep)
        val backwardOutputWrtOutputPreActivation = activation.backwardResult

        this.weighting.backwardStep(withinBatch, indexStep, backwardOutputWrtOutputPreActivation)
        val backwardOutputPreActivationWrtState = this.weighting.backwardResult

        this.bias?.backwardStep(withinBatch, indexStep, backwardOutputWrtOutputPreActivation)

        return backwardOutputPreActivationWrtState

    }

    override fun optimize(batchSize : Int) {

        if (this.unit is Optimizable) {

            this.unit.optimize(batchSize)

        }

        this.weighting.optimize(batchSize)
        this.bias?.optimize(batchSize)

    }

}