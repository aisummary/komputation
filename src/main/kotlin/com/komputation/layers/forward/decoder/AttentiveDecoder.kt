package com.komputation.layers.forward.decoder

import com.komputation.cpu.layers.combination.AdditionCombination
import com.komputation.cpu.layers.combination.additionCombination
import com.komputation.cpu.layers.forward.activation.cpuActivationLayer
import com.komputation.cpu.layers.forward.decoder.CpuAttentiveDecoder
import com.komputation.cpu.layers.forward.projection.seriesBias
import com.komputation.cpu.layers.forward.projection.seriesWeighting
import com.komputation.initialization.InitializationStrategy
import com.komputation.layers.CpuForwardLayerInstruction
import com.komputation.layers.concatenateNames
import com.komputation.layers.forward.activation.ActivationFunction
import com.komputation.layers.forward.activation.softmaxLayer
import com.komputation.layers.forward.activation.tanhLayer
import com.komputation.layers.forward.columnRepetitionLayer
import com.komputation.layers.forward.projection.weightingLayer
import com.komputation.layers.forward.transpositionLayer
import com.komputation.optimization.OptimizationInstruction


class AttentiveDecoder internal constructor(
    private val name : String?,
    private val numberSteps : Int,
    private val encodingDimension : Int,
    private val decodingDimension: Int,
    private val activationFunction: ActivationFunction,
    private val weightInitialization: InitializationStrategy,
    private val biasInitialization: InitializationStrategy?,
    private val optimization: OptimizationInstruction?) : CpuForwardLayerInstruction {

    override fun buildForCpu(): CpuAttentiveDecoder {

        val encodingWeightingName = concatenateNames(this.name, "encoding-projection")
        val encodingWeighting = weightingLayer(encodingWeightingName, this.encodingDimension, this.numberSteps, true, this.encodingDimension, this.weightInitialization, this.optimization).buildForCpu()

        val columnRepetitionLayers = Array(this.numberSteps) { indexStep ->

            val columnRepetitionLayerName = concatenateNames(this.name, "column-repetition-$indexStep")

            columnRepetitionLayer(columnRepetitionLayerName, this.encodingDimension, this.numberSteps).buildForCpu()
        }

        val attentionAdditions = Array(this.numberSteps) { indexStep ->

            val attentionAdditionName = concatenateNames(this.name, "attention-addition-$indexStep")

            additionCombination(attentionAdditionName, this.encodingDimension, this.numberSteps)

        }

        val attentionPreviousStateWeightingSeriesName = concatenateNames(this.name, "attention-previous-state-weighting")
        val attentionPreviousStateWeightingStepName = concatenateNames(this.name, "attention-previous-state-weighting-step")
        val attentionPreviousStateWeighting = seriesWeighting(attentionPreviousStateWeightingSeriesName, attentionPreviousStateWeightingStepName, this.numberSteps, true, this.decodingDimension, 1, this.encodingDimension, this.weightInitialization, this.optimization)

        val tanh = Array(this.numberSteps) { tanhLayer(this.encodingDimension, this.numberSteps).buildForCpu() }

        val scoringWeightingSeriesName = concatenateNames(this.name, "scoring-weighting")
        val scoringWeightingStepName = concatenateNames(this.name, "scoring-weighting-step")
        val scoringWeighting = seriesWeighting(scoringWeightingSeriesName, scoringWeightingStepName, this.numberSteps, false, this.encodingDimension, this.numberSteps, 1, this.weightInitialization, this.optimization)

        val softmax = Array(this.numberSteps) { softmaxLayer(1, this.numberSteps).buildForCpu() }

        val transposition = Array(this.numberSteps) { transpositionLayer(1, this.numberSteps).buildForCpu() }

        val attendedEncodingWeightingSeriesName = concatenateNames(this.name, "attended-encoding-weighting")
        val attendedEncodingWeightingStepName = concatenateNames(this.name, "attended-encoding-weighting-step")
        val attendedEncodingWeighting = seriesWeighting(attendedEncodingWeightingSeriesName, attendedEncodingWeightingStepName, this.numberSteps, false, this.encodingDimension, 1, this.encodingDimension, this.weightInitialization, this.optimization)

        val decodingPreviousStateWeightingSeriesName = concatenateNames(this.name, "decoding-previous-state-weighting")
        val decodingPreviousStateWeightingStepName = concatenateNames(this.name, "decoding-previous-state-weighting-step")
        val decodingPreviousStateWeighting = seriesWeighting(decodingPreviousStateWeightingSeriesName, decodingPreviousStateWeightingStepName, this.numberSteps, true, this.decodingDimension, 1, this.decodingDimension, this.weightInitialization, this.optimization)

        val decodingAdditions = Array(this.numberSteps) { indexStep ->

            val decodingAdditionName = concatenateNames(this.name, "decoding-addition-$indexStep")

            AdditionCombination(decodingAdditionName, this.decodingDimension, 1)

        }

        val activationName = concatenateNames(this.name, "decoding-activation")
        val activations = Array(this.numberSteps) { index ->

            cpuActivationLayer(concatenateNames(activationName, index.toString()), this.activationFunction, this.decodingDimension, 1).buildForCpu()

        }

        val bias =
            if(this.biasInitialization == null)
                null
            else {

                val biasSeriesName =  concatenateNames(this.name, "bias")
                val biasStepName =  concatenateNames(biasSeriesName, "step")

                seriesBias(biasSeriesName, biasStepName, this.numberSteps, this.decodingDimension, this.biasInitialization, this.optimization)
            }

        val attentiveDecoder = CpuAttentiveDecoder(
            this.name,
            this.numberSteps,
            this.encodingDimension,
            this.decodingDimension,
            encodingWeighting,
            attentionPreviousStateWeighting,
            columnRepetitionLayers,
            attentionAdditions,
            tanh,
            scoringWeighting,
            softmax,
            transposition,
            attendedEncodingWeighting,
            decodingPreviousStateWeighting,
            decodingAdditions,
            bias,
            activations
        )

        return attentiveDecoder

    }


}


fun attentiveDecoder(
    numberSteps : Int,
    encodingDimension : Int,
    decodingDimension: Int,
    activationFunction: ActivationFunction,
    weightInitialization: InitializationStrategy,
    biasInitialization: InitializationStrategy?,
    optimization: OptimizationInstruction?) =

    attentiveDecoder(
        null,
        numberSteps,
        encodingDimension,
        decodingDimension,
        activationFunction,
        weightInitialization,
        biasInitialization,
        optimization
    )

fun attentiveDecoder(
    name : String?,
    numberSteps : Int,
    encodingDimension : Int,
    decodingDimension: Int,
    activation: ActivationFunction,
    weightInitialization: InitializationStrategy,
    biasInitialization: InitializationStrategy?,
    optimization: OptimizationInstruction?) =

    AttentiveDecoder(
        name,
        numberSteps,
        encodingDimension,
        decodingDimension,
        activation,
        weightInitialization,
        biasInitialization,
        optimization
    )