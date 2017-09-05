package com.komputation.layers.forward

import com.komputation.cpu.layers.combination.additionCombination
import com.komputation.cpu.layers.combination.hadamardCombination
import com.komputation.cpu.layers.forward.highway.CpuHighwayLayer
import com.komputation.initialization.InitializationStrategy
import com.komputation.layers.CpuForwardLayerInstruction
import com.komputation.layers.concatenateNames
import com.komputation.layers.forward.activation.ActivationFunction
import com.komputation.layers.forward.dense.denseLayer
import com.komputation.optimization.OptimizationInstruction

class HighwayLayer internal constructor(
    private val name : String?,
    private val dimension: Int,
    private val weightInitialization: InitializationStrategy,
    private val transformationBiasInitialization: InitializationStrategy,
    private val transformationFractionBiasInitialization: InitializationStrategy,
    private val transformationFunction: ActivationFunction,
    private val optimization: OptimizationInstruction?) : CpuForwardLayerInstruction {

    override fun buildForCpu(): CpuHighwayLayer {

        val transformationName = concatenateNames(this.name, "transformation")
        val transformation = denseLayer(transformationName, this.dimension, this.dimension, this.weightInitialization, this.transformationBiasInitialization, this.transformationFunction, this.optimization).buildForCpu()

        val transformationFractionName = concatenateNames(this.name, "transformation-fraction")
        val transformationFraction = denseLayer(transformationFractionName, this.dimension, this.dimension, this.weightInitialization, this.transformationFractionBiasInitialization, ActivationFunction.Sigmoid, this.optimization).buildForCpu()

        val transformationHadamardName = concatenateNames(this.name, "transformation-hadamard")
        val transformationHadamard = hadamardCombination(transformationHadamardName, this.dimension, 1)

        val counterProbabilityName = concatenateNames(this.name, "counter-probability")
        val counterProbability = counterProbabilityLayer(counterProbabilityName, this.dimension, 1).buildForCpu()

        val carryHadamardName = concatenateNames(this.name, "carry-hadamard")
        val carryHadamard = hadamardCombination(carryHadamardName, this.dimension, 1)

        val additionName = concatenateNames(this.name, "addition")
        val addition = additionCombination(additionName, this.dimension, 1)

        val highwayLayer = CpuHighwayLayer(this.name, this.dimension, transformation, transformationFraction, transformationHadamard, counterProbability, carryHadamard, addition)

        return highwayLayer

    }

}

fun highwayLayer(
    dimension: Int,
    weightInitialization: InitializationStrategy,
    transformationBiasInitialization: InitializationStrategy,
    transformationFractionBiasInitialization: InitializationStrategy,
    transformationFunction: ActivationFunction,
    optimization: OptimizationInstruction?) =

    highwayLayer(null, dimension, weightInitialization, transformationBiasInitialization, transformationFractionBiasInitialization, transformationFunction, optimization)

fun highwayLayer(
    name : String?,
    dimension: Int,
    weightInitialization: InitializationStrategy,
    transformationBiasInitialization: InitializationStrategy,
    transformationFractionBiasInitialization: InitializationStrategy,
    transformationFunction: ActivationFunction,
    optimization: OptimizationInstruction?) =

    HighwayLayer(
        name,
        dimension,
        weightInitialization,
        transformationBiasInitialization,
        transformationFractionBiasInitialization,
        transformationFunction,
        optimization
    )