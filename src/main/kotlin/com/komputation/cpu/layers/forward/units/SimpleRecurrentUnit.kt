package com.komputation.cpu.layers.forward.units

import com.komputation.cpu.layers.combination.AdditionCombination
import com.komputation.cpu.layers.combination.additionCombination
import com.komputation.cpu.layers.forward.activation.CpuActivationLayer
import com.komputation.cpu.layers.forward.activation.cpuActivationLayer
import com.komputation.cpu.layers.forward.projection.SeriesBias
import com.komputation.cpu.layers.forward.projection.SeriesWeighting
import com.komputation.cpu.layers.forward.projection.seriesBias
import com.komputation.cpu.layers.forward.projection.seriesWeighting
import com.komputation.initialization.InitializationStrategy
import com.komputation.layers.concatenateNames
import com.komputation.layers.forward.activation.ActivationFunction
import com.komputation.optimization.Optimizable
import com.komputation.optimization.OptimizationInstruction

class SimpleRecurrentUnit internal constructor(
    name: String?,
    private val previousStateWeighting: SeriesWeighting,
    private val inputWeighting: SeriesWeighting,
    private val additions: Array<AdditionCombination>,
    private val bias: SeriesBias?,
    private val activations: Array<CpuActivationLayer>) : RecurrentUnit(name), Optimizable {

    override fun forwardStep(withinBatch : Int, indexStep: Int, state: FloatArray, input: FloatArray, isTraining : Boolean): FloatArray {

        // weighted state = state weights * state
        val weightedState = this.previousStateWeighting.forwardStep(withinBatch, indexStep, state, isTraining)

        // weighted input = input weights * input
        val weightedInput = this.inputWeighting.forwardStep(withinBatch, indexStep, input, isTraining)

        // addition = weighted input + weighted state
        val addition = this.additions[indexStep].forward(weightedState, weightedInput)

        // pre-activation = addition + bias
        val preActivation =

            if(this.bias == null) {

                addition

            }
            else {

                this.bias.forwardStep(withinBatch, indexStep, addition, isTraining)

            }

        val activation = this.activations[indexStep]

        return activation.forward(withinBatch, 1, preActivation, isTraining)

    }

    override fun backwardStep(withinBatch : Int, step : Int, chain: FloatArray): Pair<FloatArray, FloatArray> {

        // d new state / state pre-activation
        // d activate(state weights * state(1) + input weights * input(2) + bias)) / d state weights * state(1) + input weights * input(2) + bias
        val activation = this.activations[step]
        activation.backward(withinBatch, chain)
        val backwardPreActivation = activation.backwardResult

        // d state weights * state(1) + input weights * input(2) + bias / d state(1) = state weights
        this.previousStateWeighting.backwardStep(withinBatch, step, backwardPreActivation)

        // d state weights * state(1) + input weights * input(2) + bias / d input(2) = input weights
        this.inputWeighting.backwardStep(withinBatch, step, backwardPreActivation)

        // d state weights * state(1) + input weights * input(2) + bias / d bias = 1
        this.bias?.backwardStep(withinBatch, step, backwardPreActivation)

        return this.previousStateWeighting.backwardResult to this.inputWeighting.backwardResult

    }

    override fun backwardSeries() {

        this.previousStateWeighting.backwardSeries()
        this.inputWeighting.backwardSeries()

        this.bias?.backwardSeries()

    }

    override fun optimize(batchSize : Int) {

        this.previousStateWeighting.optimize(batchSize)
        this.inputWeighting.optimize(batchSize)

        this.bias?.optimize(batchSize)

    }

}

fun simpleRecurrentUnit(
    numberSteps: Int,
    inputDimension: Int,
    hiddenDimension: Int,
    stateWeightInitializationStrategy: InitializationStrategy,
    inputWeightInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    activationFunction: ActivationFunction,
    optimizationStrategy: OptimizationInstruction? = null) =

    simpleRecurrentUnit(
        null,
        numberSteps,
        inputDimension,
        hiddenDimension,
        stateWeightInitializationStrategy,
        inputWeightInitializationStrategy,
        biasInitializationStrategy,
        activationFunction,
        optimizationStrategy)

fun simpleRecurrentUnit(
    name: String?,
    numberSteps: Int,
    inputDimension: Int,
    hiddenDimension: Int,
    previousStateWeightingInitializationStrategy: InitializationStrategy,
    inputWeightingInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    activationFunction: ActivationFunction,
    optimizationStrategy: OptimizationInstruction? = null): RecurrentUnit {

    val previousStateWeightingSeriesName = concatenateNames(name, "previous-state-weighting")
    val previousStateWeightingStepName = concatenateNames(name, "previous-state-weighting-step")

    val previousStateWeighting = seriesWeighting(
        previousStateWeightingSeriesName,
        previousStateWeightingStepName,
        numberSteps,
        true,
        hiddenDimension,
        1,
        hiddenDimension,
        previousStateWeightingInitializationStrategy,
        optimizationStrategy)

    val inputWeightingSeriesName = concatenateNames(name, "input-weighting")
    val inputWeightingStepName = concatenateNames(name, "input-weighting-step")

    val inputWeighting = seriesWeighting(
        inputWeightingSeriesName,
        inputWeightingStepName,
        numberSteps,
        false,
        inputDimension,
        1,
        hiddenDimension,
        inputWeightingInitializationStrategy,
        optimizationStrategy)

    val additions = Array(numberSteps) { indexStep ->

        val additionName = concatenateNames(name, "addition-step-$indexStep")
        additionCombination(additionName, hiddenDimension, 1)

    }

    val bias =

        if(biasInitializationStrategy == null)
            null
        else {

            val biasSeriesName = concatenateNames(name, "bias")
            val biasStepName = concatenateNames(biasSeriesName, "step")

            seriesBias(biasSeriesName, biasStepName, numberSteps, hiddenDimension, biasInitializationStrategy, optimizationStrategy)

        }

    val activationName = concatenateNames(name, "activation")
    val activationLayers = Array(numberSteps) { index ->

        cpuActivationLayer(concatenateNames(activationName, index.toString()), activationFunction, hiddenDimension, 1).buildForCpu()

    }

    val unitName = concatenateNames(name, "unit")
    val unit = SimpleRecurrentUnit(unitName, previousStateWeighting, inputWeighting, additions, bias, activationLayers)

    return unit

}