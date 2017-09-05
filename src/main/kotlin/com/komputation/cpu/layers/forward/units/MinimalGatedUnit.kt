package com.komputation.cpu.layers.forward.units

import com.komputation.cpu.layers.combination.AdditionCombination
import com.komputation.cpu.layers.combination.HadamardCombination
import com.komputation.cpu.layers.combination.additionCombination
import com.komputation.cpu.layers.combination.hadamardCombination
import com.komputation.cpu.layers.forward.CpuCounterProbabilityLayer
import com.komputation.cpu.layers.forward.activation.CpuActivationLayer
import com.komputation.cpu.layers.forward.projection.seriesBias
import com.komputation.cpu.layers.forward.projection.seriesWeighting
import com.komputation.cpu.optimization.DenseAccumulator
import com.komputation.initialization.InitializationStrategy
import com.komputation.layers.concatenateNames
import com.komputation.layers.forward.activation.sigmoidLayer
import com.komputation.layers.forward.counterProbabilityLayer
import com.komputation.optimization.Optimizable
import com.komputation.optimization.OptimizationInstruction

class MinimalGatedUnit internal constructor(
    name: String?,
    private val inputDimension: Int,
    private val hiddenDimension: Int,
    private val forgetUnit: SimpleRecurrentUnit,
    private val shortTermResponse: ShortTermResponse,
    private val counterProbabilities: Array<CpuCounterProbabilityLayer>,
    private val longTermHadamards: Array<HadamardCombination>,
    private val shortTermHadamards: Array<HadamardCombination>,
    private val stateAdditions: Array<AdditionCombination>) : RecurrentUnit(name), Optimizable {

    private val previousStateAccumulator = DenseAccumulator(this.hiddenDimension)
    private val inputAccumulator = DenseAccumulator(this.inputDimension)

    override fun forwardStep(withinBatch : Int, indexStep: Int, state : FloatArray, input : FloatArray, isTraining : Boolean): FloatArray {

        val forget = this.forgetUnit.forwardStep(withinBatch, indexStep, state, input, isTraining)

        // 1 - forget
        val counterProbabilityLayer = this.counterProbabilities[indexStep]
        val oneMinusForget = counterProbabilityLayer.forward(withinBatch, 1, forget, isTraining)

        val longTermHadamard = this.longTermHadamards[indexStep]

        // The long-term component is the state multiplied by the counter-probability.
        val longTermComponent = longTermHadamard.forward(oneMinusForget, state)

        val shortTermResponse = this.shortTermResponse.forward(withinBatch, indexStep, state, input, forget, isTraining)

        // The short-term component is the short-term response multiplied by forget
        val shortTermComponent = this.shortTermHadamards[indexStep].forward(forget, shortTermResponse)

        // The new state is the sum of the long-term component and the short-term component.
        return this.stateAdditions[indexStep].forward(longTermComponent, shortTermComponent)

    }


    private fun backwardLongTermComponent(withinBatch : Int, step: Int, backwardChainWrtLongTermComponent: FloatArray) {

        // (1 - forget) (.) previous state / d (1 - forget) = previous state
        val backwardLongTermComponentWrtKeep = this.longTermHadamards[step].backwardFirst(backwardChainWrtLongTermComponent)

        // d (1 - forget) / d forget = -1
        val counterProbability = this.counterProbabilities[step]
        counterProbability.backward(withinBatch, backwardLongTermComponentWrtKeep)
        val backwardKeepWrtForget = counterProbability.backwardResult
        val (backwardForgetWrtPreviousState, backwardForgetWrtInput) = this.forgetUnit.backwardStep(withinBatch, step, backwardKeepWrtForget)

        this.previousStateAccumulator.accumulate(backwardForgetWrtPreviousState)
        this.inputAccumulator.accumulate(backwardForgetWrtInput)

        // (1 - forget) (.) previous state / d previous state = (1 - forget)
        val backwardLongTermComponentWrtPreviousState = this.longTermHadamards[step].backwardSecond(backwardChainWrtLongTermComponent)
        this.previousStateAccumulator.accumulate(backwardLongTermComponentWrtPreviousState)
    }

    private fun backwardShortTermComponent(withinBatch : Int, step: Int, backChainWrtShortTermComponent: FloatArray) {

        // short-term component = forget (.) short-term response

        // d short-term component / forget = short-term response
        val backwardShortTermComponentWrtForget = this.shortTermHadamards[step].backwardFirst(backChainWrtShortTermComponent)

        // d forget / d previous state, d forget / input
        val (backwardShortTermComponentForgetWrtPreviousState, backwardShortTermComponentForgetWrtInput) = this.forgetUnit.backwardStep(withinBatch, step, backwardShortTermComponentWrtForget)
        this.previousStateAccumulator.accumulate(backwardShortTermComponentForgetWrtPreviousState)
        this.inputAccumulator.accumulate(backwardShortTermComponentForgetWrtInput)

        // d short-term component / short-term response = forget
        val backwardShortTermComponentWrtShortTermResponse = this.shortTermHadamards[step].backwardSecond(backChainWrtShortTermComponent)

        val (backwardShortTermMemoryWrtForget, shortTermResponsePair) = this.shortTermResponse.backward(withinBatch, step, backwardShortTermComponentWrtShortTermResponse)
        val (backwardShortTermMemoryWrtPreviousState, backwardShortTermWeightedInputWrtInput) = shortTermResponsePair

        // d forget / d previous state, d forget / input
        val (backwardShortTermMemoryForgetWrtPreviousState, backwardShortTermMemoryForgetWrtInput) = this.forgetUnit.backwardStep(withinBatch, step, backwardShortTermMemoryWrtForget)
        this.previousStateAccumulator.accumulate(backwardShortTermMemoryForgetWrtPreviousState)
        this.inputAccumulator.accumulate(backwardShortTermMemoryForgetWrtInput)

        this.previousStateAccumulator.accumulate(backwardShortTermMemoryWrtPreviousState)
        this.inputAccumulator.accumulate(backwardShortTermWeightedInputWrtInput)

    }

    private val previousStateAccumulation = FloatArray(this.hiddenDimension)
    private val inputAccumulation = FloatArray(this.inputDimension)

    override fun backwardStep(withinBatch: Int, step : Int, chain : FloatArray): Pair<FloatArray, FloatArray> {

        // d (long-term component + short-term component) / d long-term component
        val backChainWrtLongTermComponent = this.stateAdditions[step].backwardFirst(chain)
        this.backwardLongTermComponent(withinBatch, step, backChainWrtLongTermComponent)

        // d (long-term component + short-term component) / d short-term component
        val backChainWrtShortTermComponent = this.stateAdditions[step].backwardSecond(chain)
        this.backwardShortTermComponent(withinBatch, step, backChainWrtShortTermComponent)

        System.arraycopy(this.previousStateAccumulator.getAccumulation(), 0, this.previousStateAccumulation, 0, this.hiddenDimension)
        System.arraycopy(this.inputAccumulator.getAccumulation(), 0, this.inputAccumulation, 0, this.inputDimension)

        this.inputAccumulator.reset()
        this.previousStateAccumulator.reset()

        return previousStateAccumulation to inputAccumulation

    }

    override fun backwardSeries() {

        this.forgetUnit.backwardSeries()
        this.shortTermResponse.backwardSeries()

    }

    override fun optimize(batchSize : Int) {

        this.forgetUnit.optimize(batchSize)
        this.shortTermResponse.optimize(batchSize)

    }

}

fun minimalGatedUnit(
    numberSteps: Int,
    inputDimension: Int,
    hiddenDimension: Int,
    forgetPreviousStateWeightInitializationStrategy: InitializationStrategy,
    forgetInputWeightInitializationStrategy: InitializationStrategy,
    forgetBiasInitializationStrategy: InitializationStrategy?,
    shortTermMemoryWeightInitializationStrategy: InitializationStrategy,
    shortTermInputWeightInitializationStrategy: InitializationStrategy,
    shortTermBiasInitializationStrategy: InitializationStrategy?,
    optimization: OptimizationInstruction? = null) =

    minimalGatedUnit(
        null,
        numberSteps,
        inputDimension,
        hiddenDimension,
        forgetPreviousStateWeightInitializationStrategy,
        forgetInputWeightInitializationStrategy,
        forgetBiasInitializationStrategy,
        shortTermMemoryWeightInitializationStrategy,
        shortTermInputWeightInitializationStrategy,
        shortTermBiasInitializationStrategy,
        optimization)

fun minimalGatedUnit(
    name: String?,
    numberSteps: Int,
    inputDimension: Int,
    hiddenDimension: Int,
    forgetPreviousStateWeightInitializationStrategy: InitializationStrategy,
    forgetInputWeightInitializationStrategy: InitializationStrategy,
    forgetBiasInitializationStrategy: InitializationStrategy?,
    shortTermMemoryWeightInitializationStrategy: InitializationStrategy,
    shortTermInputWeightInitializationStrategy: InitializationStrategy,
    shortTermBiasInitializationStrategy: InitializationStrategy?,
    optimization: OptimizationInstruction? = null): RecurrentUnit {

    val forgetPreviousStateWeightingSeriesName = concatenateNames(name, "forget-previous-state-weighting")
    val forgetPreviousStateWeightingStepName = concatenateNames(name, "forget-previous-state-weighting-step")
    val forgetPreviousStateWeighting = seriesWeighting(forgetPreviousStateWeightingSeriesName, forgetPreviousStateWeightingStepName, numberSteps, true, hiddenDimension, 1, hiddenDimension, forgetPreviousStateWeightInitializationStrategy, optimization)

    val forgetInputWeightingSeriesName = concatenateNames(name, "forget-input-weighting")
    val forgetInputWeightingStepName = concatenateNames(name, "forget-input-weighting-step")
    val forgetInputWeighting = seriesWeighting(forgetInputWeightingSeriesName, forgetInputWeightingStepName, numberSteps, false, inputDimension, 1, hiddenDimension, forgetInputWeightInitializationStrategy, optimization)

    val forgetAdditions = Array(numberSteps) { indexStep ->

        val forgetAdditionName = concatenateNames(name, "forget-addition-step-$indexStep")
        additionCombination(forgetAdditionName, hiddenDimension, 1)

    }

    val forgetBias =

        if (forgetBiasInitializationStrategy != null) {

            val forgetBiasSeriesName = concatenateNames(name, "forget-bias")
            val forgetBiasStepName = concatenateNames(forgetBiasSeriesName, "step")
            seriesBias(forgetBiasSeriesName, forgetBiasStepName, numberSteps, hiddenDimension, forgetBiasInitializationStrategy, optimization)

        }
        else {

            null

        }

    val forgetActivations = Array<CpuActivationLayer>(numberSteps) { indexStep ->

        val forgetActivationName = concatenateNames(name, "forget-activation-step-$indexStep")

        sigmoidLayer(forgetActivationName, hiddenDimension).buildForCpu()

    }

    val forgetUnit = SimpleRecurrentUnit(name, forgetPreviousStateWeighting, forgetInputWeighting, forgetAdditions, forgetBias, forgetActivations)

    val shortTermResponse = shortTermResponse("short-term-response", numberSteps, hiddenDimension, inputDimension, shortTermMemoryWeightInitializationStrategy, shortTermInputWeightInitializationStrategy, shortTermBiasInitializationStrategy, optimization)

    val keepSubtractions = Array(numberSteps) { indexStep ->

        val keepSubtractionName = concatenateNames(name, "keep-subtraction-$indexStep")
        counterProbabilityLayer(keepSubtractionName, hiddenDimension, 1).buildForCpu()

    }

    val shortTermHadamards = Array(numberSteps) { indexStep ->

        val shortTermHadamardName = concatenateNames(name, "short-term-hadamard-step-$indexStep")
        hadamardCombination(shortTermHadamardName, hiddenDimension, 1)

    }

    val longTermHadamards = Array(numberSteps) { indexStep ->

        val shortTermHadamardName = concatenateNames(name, "long-term-hadamard-step-$indexStep")
        hadamardCombination(shortTermHadamardName, hiddenDimension, 1)

    }

    val stateAdditions = Array(numberSteps) { indexStep ->

        val shortTermAdditionName = concatenateNames(name, "state-addition-step-$indexStep")
        additionCombination(shortTermAdditionName, hiddenDimension, 1)

    }

    val minimalGatedUnit = MinimalGatedUnit(
        name,
        inputDimension,
        hiddenDimension,
        forgetUnit,
        shortTermResponse,
        keepSubtractions,
        shortTermHadamards,
        longTermHadamards,
        stateAdditions)

    return minimalGatedUnit

}