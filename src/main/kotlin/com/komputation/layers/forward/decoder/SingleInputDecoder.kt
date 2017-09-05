package com.komputation.layers.forward.decoder

import com.komputation.cpu.layers.forward.activation.cpuActivationLayer
import com.komputation.cpu.layers.forward.decoder.CpuSingleInputDecoder
import com.komputation.cpu.layers.forward.projection.seriesBias
import com.komputation.cpu.layers.forward.projection.seriesWeighting
import com.komputation.cpu.layers.forward.units.RecurrentUnit
import com.komputation.initialization.InitializationStrategy
import com.komputation.layers.CpuForwardLayerInstruction
import com.komputation.layers.concatenateNames
import com.komputation.layers.forward.activation.ActivationFunction
import com.komputation.optimization.OptimizationInstruction

class SingleInputDecoder internal constructor(
    private val name : String?,
    private val numberSteps: Int,
    private val hiddenDimension : Int,
    private val outputDimension: Int,
    private val unit : RecurrentUnit,
    private val weightInitialization: InitializationStrategy,
    private val biasInitialization: InitializationStrategy?,
    private val activationFunction: ActivationFunction,
    private val optimization: OptimizationInstruction?): CpuForwardLayerInstruction {

    override fun buildForCpu(): CpuSingleInputDecoder {

        val weightingSeriesName = concatenateNames(this.name, "weighting")
        val weightingStepName = concatenateNames(weightingSeriesName, "weighting-step")
        val weighting = seriesWeighting(weightingSeriesName, weightingStepName, this.numberSteps, false, this.hiddenDimension, 1, this.outputDimension, this.weightInitialization, optimization)

        val bias =

            if (this.biasInitialization != null) {

                val biasSeriesName = concatenateNames(this.name, "bias")
                val biasStepName = concatenateNames(biasSeriesName, "step")
                seriesBias(biasSeriesName, biasStepName, this.numberSteps, this.outputDimension, this.biasInitialization, this.optimization)

            }
            else {

                null

            }

        val activationName = concatenateNames(this.name, "activation")
        val activations = Array(this.numberSteps) { index ->

            cpuActivationLayer(concatenateNames(activationName, index.toString()), this.activationFunction, this.outputDimension, 1).buildForCpu()

        }

        val decoder = CpuSingleInputDecoder(
            this.name,
            this.numberSteps,
            this.hiddenDimension,
            this.outputDimension,
            this.unit,
            weighting,
            bias,
            activations)

        return decoder

    }

}

fun singleInputDecoder(
    numberSteps: Int,
    hiddenDimension : Int,
    outputDimension: Int,
    unit : RecurrentUnit,
    weightInitialization: InitializationStrategy,
    biasInitialization: InitializationStrategy?,
    activation: ActivationFunction,
    optimization: OptimizationInstruction?) =

    singleInputDecoder(
        null,
        numberSteps,
        hiddenDimension,
        outputDimension,
        unit,
        weightInitialization,
        biasInitialization,
        activation,
        optimization)


fun singleInputDecoder(
    name : String?,
    numberSteps: Int,
    hiddenDimension : Int,
    outputDimension: Int,
    unit : RecurrentUnit,
    weightInitialization: InitializationStrategy,
    biasInitialization: InitializationStrategy?,
    activation: ActivationFunction,
    optimization: OptimizationInstruction?) =

    SingleInputDecoder(
        name,
        numberSteps,
        hiddenDimension,
        outputDimension,
        unit,
        weightInitialization,
        biasInitialization,
        activation,
        optimization)