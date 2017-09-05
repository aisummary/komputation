package com.komputation.cpu.layers.forward.decoder

import com.komputation.cpu.functions.add
import com.komputation.cpu.functions.getStep
import com.komputation.cpu.layers.BaseCpuForwardLayer
import com.komputation.cpu.layers.CpuLayerState
import com.komputation.cpu.layers.forward.activation.CpuActivationLayer
import com.komputation.cpu.layers.forward.projection.SeriesBias
import com.komputation.cpu.layers.forward.projection.SeriesWeighting
import com.komputation.cpu.layers.forward.units.RecurrentUnit
import com.komputation.layers.Resourceful
import com.komputation.optimization.Optimizable


// The first input is empty.
// Starting with the second input, the input at step t is the output of step t-1.
class CpuSingleInputDecoder internal constructor(
    name : String?,
    private val numberSteps : Int,
    private val hiddenDimension : Int,
    private val outputDimension : Int,
    private val unit : RecurrentUnit,
    private val weighting: SeriesWeighting,
    private val bias: SeriesBias?,
    private val activations: Array<CpuActivationLayer>) : BaseCpuForwardLayer(name), Resourceful, Optimizable {

    private var inputs = emptyArray<FloatArray>()
    private var states = emptyArray<FloatArray>()

    override val numberOutputRows = this.outputDimension
    override val numberOutputColumns = this.numberSteps
    override var forwardResult = FloatArray(0)

    override val numberInputRows = this.hiddenDimension
    override val numberInputColumns = this.numberSteps
    override var backwardResult = FloatArray(0)

    override fun acquire(maximumBatchSize: Int) {

        this.inputs = Array(this.numberSteps+1) { FloatArray(this.outputDimension) }
        this.states = Array(this.numberSteps+1) { FloatArray(this.hiddenDimension) }

        this.forwardResult = FloatArray(this.numberSteps * this.outputDimension)
        this.backwardResult = FloatArray(this.numberSteps * this.hiddenDimension)

    }

    override fun release() {

    }

    override fun forward(withinBatch : Int, numberInputColumns: Int, input: FloatArray, isTraining : Boolean): FloatArray {

        // Use the encoder output as the first state
        this.states[0] = input

        for (indexStep in 0..this.numberSteps - 1) {

            val newState = this.unit.forwardStep(withinBatch, indexStep, this.states[indexStep], this.inputs[indexStep], isTraining)
            this.states[indexStep+1] = newState

            val output = this.forwardOutput(withinBatch, indexStep, newState, isTraining)
            this.inputs[indexStep+1] = output

            System.arraycopy(output, 0, this.forwardResult, indexStep * this.outputDimension, this.outputDimension)

        }

        return this.forwardResult

    }

    private fun forwardOutput(withinBatch : Int, indexStep: Int, newState: FloatArray, isTraining: Boolean): FloatArray {

        this.weighting.forwardStep(withinBatch, indexStep,  newState, isTraining)

        val biased : CpuLayerState =

            if (this.bias != null) {

                this.bias.forwardStep(withinBatch, indexStep, this.weighting.forwardResult, isTraining)
                this.bias

            }
            else {

                this.weighting

            }

        return this.activations[indexStep].forward(withinBatch, biased.numberOutputColumns, biased.forwardResult, isTraining)

    }

    private val chainStep = FloatArray(this.outputDimension)
    private val backwardPreActivationSum = FloatArray(this.hiddenDimension)

    // Incoming gradient: d chain / d series prediction
    override fun backward(withinBatch: Int, chain: FloatArray): FloatArray {

        // Differentiate the chain w.r.t. input
        var backwardPreActivationWrtInput : FloatArray? = null

        // Differentiate the chain w.r.t previous state.
        // This is done at each step. For the first step (t=1), the chain is differentiated w.r.t. to the initial state (t=0).
        var backwardPreActivationWrtPreviousState : FloatArray? = null

        for (indexStep in this.numberSteps - 1 downTo 0) {

            getStep(chain, indexStep, this.chainStep, this.outputDimension)

            val backwardOutputPreActivationWrtState = backwardOutput(withinBatch, indexStep, this.chainStep, backwardPreActivationWrtInput)

            if (backwardPreActivationWrtPreviousState != null) {

                add(backwardPreActivationWrtPreviousState, backwardOutputPreActivationWrtState, backwardPreActivationSum, this.hiddenDimension)

            }
            else {

                System.arraycopy(backwardOutputPreActivationWrtState, 0, this.backwardPreActivationSum, 0, this.hiddenDimension)

            }

            val (newBackwardStatePreActivationWrtPreviousState, newBackwardStatePreActivationWrtInput) = this.unit.backwardStep(withinBatch, indexStep, this.backwardPreActivationSum)

            backwardPreActivationWrtInput = newBackwardStatePreActivationWrtInput
            backwardPreActivationWrtPreviousState = newBackwardStatePreActivationWrtPreviousState

        }

        this.unit.backwardSeries()

        this.weighting.backwardSeries()
        this.bias?.backwardSeries()

        this.backwardResult = backwardPreActivationWrtPreviousState!!

        return this.backwardResult
    }

    private val outputSum = FloatArray(this.outputDimension)

    private fun backwardOutput(withinBatch: Int, indexStep: Int, chainStep: FloatArray, backwardStatePreActivationWrtInput: FloatArray?): FloatArray {

        // The input gradient for step t+1 is added to the chain step t
        val addition = if (backwardStatePreActivationWrtInput != null) {

            // d chain / d output(index+1) * d output(index+1) / d input(index + 1) *  d input(index + 1) / d output(index)
            add(backwardStatePreActivationWrtInput, chainStep, this.outputSum, this.outputDimension)

            this.outputSum

        }
        // except in the case of the last step (t = T)
        else {

            chainStep

        }

        val activation = this.activations[indexStep]

        activation.backward(withinBatch, addition)
        val backwardOutputWrtOutputPreActivation = activation.backwardResult

        this.bias?.backwardStep(withinBatch, indexStep, backwardOutputWrtOutputPreActivation)

        this.weighting.backwardStep(withinBatch, indexStep, backwardOutputWrtOutputPreActivation)
        val backwardOutputPreActivationWrtState = this.weighting.backwardResult

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