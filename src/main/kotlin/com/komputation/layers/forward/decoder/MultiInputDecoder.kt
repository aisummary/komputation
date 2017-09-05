package com.komputation.layers.forward.decoder

import com.komputation.cpu.layers.forward.activation.cpuActivationLayer
import com.komputation.cpu.layers.forward.decoder.CpuMultiInputDecoder
import com.komputation.cpu.layers.forward.projection.seriesBias
import com.komputation.cpu.layers.forward.projection.seriesWeighting
import com.komputation.cpu.layers.forward.units.RecurrentUnit
import com.komputation.initialization.InitializationStrategy
import com.komputation.layers.CpuForwardLayerInstruction
import com.komputation.layers.concatenateNames
import com.komputation.layers.forward.activation.ActivationFunction
import com.komputation.optimization.OptimizationInstruction

class MultiInputDecoder internal constructor(
    private val name : String?,
    private val numberSteps: Int,
    private val inputDimension: Int,
    private val hiddenDimension: Int,
    private val outputDimension: Int,
    private val unit : RecurrentUnit,
    private val weightInitialization: InitializationStrategy,
    private val biasInitialization: InitializationStrategy?,
    private val activationFunction: ActivationFunction,
    private val optimization: OptimizationInstruction?) : CpuForwardLayerInstruction {

    override fun buildForCpu(): CpuMultiInputDecoder {

        val weightingSeriesName = concatenateNames(this.name, "weighting")
        val weightingStepName = concatenateNames(this.name, "weighting-step")
        val weighting = seriesWeighting(weightingSeriesName, weightingStepName, this.numberSteps, false, this.hiddenDimension, 1, this.outputDimension, this.weightInitialization, this.optimization)

        val bias =
            if (this.biasInitialization != null) {

                val biasSeriesName = concatenateNames(this.name, "bias")
                val biasStepName = concatenateNames(this.name, "step")
                seriesBias(biasSeriesName, biasStepName, this.numberSteps, this.outputDimension, this.biasInitialization, this.optimization)

            }
            else {

                null

            }

        val activationName = concatenateNames(this.name, "activation")
        val activations = Array(this.numberSteps) { index ->

            cpuActivationLayer(concatenateNames(activationName, index.toString()), this.activationFunction, this.outputDimension, this.numberSteps).buildForCpu()

        }

        return CpuMultiInputDecoder(
            this.name,
            this.numberSteps,
            this.inputDimension,
            this.hiddenDimension,
            this.outputDimension,
            unit,
            weighting,
            bias,
            activations)

    }

}

fun multiInputDecoder(
    numberSteps: Int,
    inputDimension: Int,
    hiddenDimension: Int,
    outputDimension: Int,
    unit : RecurrentUnit,
    weightInitialization: InitializationStrategy,
    biasInitialization: InitializationStrategy?,
    activation: ActivationFunction,
    optimization: OptimizationInstruction?) =

    multiInputDecoder(
        null,
        numberSteps,
        inputDimension,
        hiddenDimension,
        outputDimension,
        unit,
        weightInitialization,
        biasInitialization,
        activation,
        optimization)

fun multiInputDecoder(
    name : String?,
    numberSteps: Int,
    inputDimension: Int,
    hiddenDimension: Int,
    outputDimension: Int,
    unit : RecurrentUnit,
    weightInitialization: InitializationStrategy,
    biasInitialization: InitializationStrategy?,
    activation: ActivationFunction,
    optimization: OptimizationInstruction?) =

    MultiInputDecoder(
        name,
        numberSteps,
        inputDimension,
        hiddenDimension,
        outputDimension,
        unit,
        weightInitialization,
        biasInitialization,
        activation,
        optimization)