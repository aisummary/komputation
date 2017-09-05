package com.komputation.cpu.layers.forward.decoder

import com.komputation.cpu.functions.*
import com.komputation.cpu.layers.BaseCpuForwardLayer
import com.komputation.cpu.layers.combination.AdditionCombination
import com.komputation.cpu.layers.forward.CpuColumnRepetitionLayer
import com.komputation.cpu.layers.forward.CpuTranspositionLayer
import com.komputation.cpu.layers.forward.activation.CpuActivationLayer
import com.komputation.cpu.layers.forward.activation.CpuSoftmaxLayer
import com.komputation.cpu.layers.forward.activation.CpuTanhLayer
import com.komputation.cpu.layers.forward.projection.CpuWeightingLayer
import com.komputation.cpu.layers.forward.projection.SeriesBias
import com.komputation.cpu.layers.forward.projection.SeriesWeighting
import com.komputation.cpu.optimization.DenseAccumulator
import com.komputation.layers.Resourceful
import com.komputation.optimization.Optimizable
import java.util.*

class CpuAttentiveDecoder internal constructor(
    name : String?,
    private val numberSteps : Int,
    private val encodingDimension: Int,
    private val decodingDimension: Int,
    private val encodingWeighting: CpuWeightingLayer,
    private val attentionPreviousStateWeighting: SeriesWeighting,
    private val columnRepetitions: Array<CpuColumnRepetitionLayer>,
    private val attentionAdditions : Array<AdditionCombination>,
    private val tanh: Array<CpuTanhLayer>,
    private val scoringWeighting: SeriesWeighting,
    private val softmax : Array<CpuSoftmaxLayer>,
    private val transposition: Array<CpuTranspositionLayer>,
    private val attendedEncodingWeighting: SeriesWeighting,
    private val decodingPreviousDecoderWeighting: SeriesWeighting,
    private val decodingAdditions: Array<AdditionCombination>,
    private val bias : SeriesBias?,
    private val activations: Array<CpuActivationLayer>) : BaseCpuForwardLayer(name), Resourceful, Optimizable {

    private var attentionDistribution = FloatArray(0)

    private val inputSize = this.numberSteps * this.encodingDimension
    private var input = FloatArray(0)
    private val inputAccumulator = DenseAccumulator(this.inputSize)

    override val numberOutputRows = this.decodingDimension
    override val numberOutputColumns = this.numberSteps
    override var forwardResult = FloatArray(0)

    override val numberInputRows = this.encodingDimension
    override val numberInputColumns = this.numberSteps
    private val numberBackwardEntries = this.numberInputRows * numberInputColumns
    override var backwardResult = FloatArray(0)

    private var previousDecoderState = FloatArray(this.decodingDimension)

    private var backwardSumWrtDecoderState = FloatArray(0)
    private var backwardOutputWrtDecoderState = FloatArray(0)

    private var backwardAttendedEncodingWrtEncoding = FloatArray(0)
    private var backwardWeightedAttendedEncodingWrtTransposedAttentionDistribution = FloatArray(0)

    override fun acquire(maximumBatchSize: Int) {

        this.attentionDistribution = FloatArray(this.numberSteps)

        this.input = FloatArray(this.inputSize)

        this.forwardResult = FloatArray(this.numberSteps * this.decodingDimension)

        this.backwardResult = FloatArray(this.numberBackwardEntries)

        this.previousDecoderState = FloatArray(this.decodingDimension)

        this.backwardSumWrtDecoderState = FloatArray(this.decodingDimension)
        this.backwardOutputWrtDecoderState = FloatArray(this.decodingDimension)

        this.backwardAttendedEncodingWrtEncoding = FloatArray(this.inputSize)
        this.backwardWeightedAttendedEncodingWrtTransposedAttentionDistribution = FloatArray(this.numberSteps)

    }

    override fun release() {

    }

    override fun forward(withinBatch : Int, numberInputColumns : Int, input : FloatArray, isTraining : Boolean): FloatArray {

        this.input = input

        // encoding weights * encodings
        val projectedEncoding = this.encodingWeighting.forward(withinBatch, numberInputColumns, input, isTraining)

        val blasEncodingMatrix = org.jblas.FloatMatrix(this.encodingDimension, this.numberSteps, *this.input)

        Arrays.fill(this.previousDecoderState, 0f)

        for (indexStep in 0..this.numberSteps - 1) {

            // previous decoder state weights (for attention) * previous decoder state
            val attentionWeightedPreviousState = this.attentionPreviousStateWeighting.forwardStep(withinBatch, indexStep, this.previousDecoderState, isTraining)

            // expanded weighted previous decoder state (for attention)
            val expandedAttentionWeightedPreviousState = this.columnRepetitions[indexStep].forward(withinBatch, 1, attentionWeightedPreviousState, isTraining)

            // pre-activation = projected encodings + expanded weighted previous decoder state (for attention)
            val attentionPreActivation = this.attentionAdditions[indexStep].forward(projectedEncoding, expandedAttentionWeightedPreviousState)

            // attention activation = tanh(pre-activation)
            val attentionActivation = this.tanh[indexStep].forward(withinBatch, numberInputColumns, attentionPreActivation, isTraining)

            // unnormalized scores = scoring weights * attention activation << row vector
            this.scoringWeighting.forwardStep(withinBatch, indexStep, attentionActivation, isTraining)

            // normalized scores = row-wise softmax (unnormalized attention scores) << row vector
            this.attentionDistribution = this.softmax[indexStep].forward(withinBatch, numberInputColumns, this.scoringWeighting.forwardResult, isTraining)

            // normalized scores as a column vector = transposed(normalized scores as a row vector)
            val transposedAttentionDistribution = this.transposition[indexStep].forward(withinBatch, numberInputColumns, attentionDistribution, isTraining)

            // attended encoding = encodings * normalized scores as column vector
            val blasTransposedAttentionDistribution = org.jblas.FloatMatrix(this.numberSteps, 1, *transposedAttentionDistribution)

            val blasAttendedEncoding = org.jblas.FloatMatrix(this.encodingDimension, 1)

            multiply(
                blasEncodingMatrix,
                blasTransposedAttentionDistribution,
                blasAttendedEncoding
            )

            val attendedEncoding = blasAttendedEncoding.data

            // weighted attended encoding = attended encoding weights * attended encoding
            val weightedAttendedEncoding = this.attendedEncodingWeighting.forwardStep(withinBatch, indexStep, attendedEncoding, isTraining)

            // previous decoder state weights (for decoding) * previous decoder state weights
            val decodingWeightedPreviousState = this.decodingPreviousDecoderWeighting.forwardStep(withinBatch, indexStep, this.previousDecoderState, isTraining)

            // weighted attended encoding + decoding weighted previous state
            val decodingAddition = this.decodingAdditions[indexStep].forward(weightedAttendedEncoding, decodingWeightedPreviousState)

            val newDecoderStatePreActivation =

                if(this.bias == null) {

                    decodingAddition

                }
                else {

                    this.bias.forwardStep(withinBatch, indexStep, decodingAddition, isTraining)

                }

            val newDecoderState = this.activations[indexStep].forward(withinBatch, 1, newDecoderStatePreActivation, isTraining)

            setStep(newDecoderState, indexStep, this.forwardResult, this.decodingDimension)

            this.previousDecoderState = newDecoderState

        }

        return this.forwardResult

    }

    override fun backward(withinBatch : Int, chain: FloatArray): FloatArray {

        var backwardNextDecoderStateWrtDecoderState : FloatArray? = null

        for (indexStep in this.numberSteps - 1 downTo 0) {

            val isLastStep = indexStep + 1 == this.numberSteps

            getStep(chain, indexStep, this.backwardOutputWrtDecoderState, this.decodingDimension)

            if (isLastStep) {

                this.backwardSumWrtDecoderState = this.backwardOutputWrtDecoderState

            }
            else {

                add(this.backwardOutputWrtDecoderState, backwardNextDecoderStateWrtDecoderState!!, this.backwardSumWrtDecoderState, this.decodingDimension)

            }

            // f'(U_a * Ea^T + U_d * d_t-1 + bias) = d f(U_a * Ea^T + U_d * d_t-1 + bias) / d (U_a * Ea^T + U_d * d_t-1 + bias)
            val activation = this.activations[indexStep]
            activation.backward(withinBatch, this.backwardSumWrtDecoderState)
            val backwardDecodingWrtDecodingPreActivation = activation.backwardResult

            // d (U_a * Ea^T + U_d * d_t-1 + bias) / d Ea^T
            this.attendedEncodingWeighting.backwardStep(withinBatch, indexStep, backwardDecodingWrtDecodingPreActivation)
            val backwardPreActivationWrtWeightedAttendedEncoding = this.attendedEncodingWeighting.backwardResult

            // d (U_a * Ea^T + U_d * d_t-1 + bias) / d d_t-1
            this.decodingPreviousDecoderWeighting.backwardStep(withinBatch, indexStep, backwardDecodingWrtDecodingPreActivation)
            val backwardPreActivationWrtWeightedPreviousStateForDecoding = this.decodingPreviousDecoderWeighting.backwardResult

            this.bias?.backwardStep(withinBatch, indexStep, backwardDecodingWrtDecodingPreActivation)

            /* Ea^T
                                        a_1
                                        a_2
                                        a_3
                e(1)_1 e(2)_1 e(3)_1    e(1)_1 * a_1 + e(2)_1 * a_2 + e(3)_1 * a_3
                e(1)_2 e(2)_2 e(3)_2    e(1)_2 * a_1 + e(2)_2 * a_2 + e(3)_2 * a_3*/

            /* d Ea^t / d a^t
               d Ea^t / d a_(1) = e(1)_1 + e(1)_2
               d Ea^t / d a_(2) = e(2)_1 + e(2)_2
               d Ea^t / d a_(3) = e(3)_1 + e(3)_2*/

            val backwardPreActivationWrtWeightedAttendedEncodingNumberRows = this.attendedEncodingWeighting.numberInputRows
            val backwardPreActivationWrtWeightedAttendedEncodingNumberColumns = this.attendedEncodingWeighting.numberInputColumns

            backwardProjectionWrtInput(
                this.numberSteps,
                1,
                this.input,
                this.encodingDimension,
                backwardPreActivationWrtWeightedAttendedEncoding,
                backwardPreActivationWrtWeightedAttendedEncodingNumberRows,
                this.backwardWeightedAttendedEncodingWrtTransposedAttentionDistribution)

            /* d Ea^t / d E
               d Ea^t / d e(1)_1 = a_1
               d Ea^t / d e(1)_2 = a_1 */

            backwardProjectionWrtWeights(
                this.encodingDimension,
                this.numberSteps,
                this.attentionDistribution,
                this.numberSteps,
                backwardPreActivationWrtWeightedAttendedEncoding,
                backwardPreActivationWrtWeightedAttendedEncodingNumberRows,
                backwardPreActivationWrtWeightedAttendedEncodingNumberColumns,
                this.backwardAttendedEncodingWrtEncoding)

            this.inputAccumulator.accumulate(this.backwardAttendedEncodingWrtEncoding)

            // d a^T / d a = d a^T / d softmax(pre-activation)
            val transposition = this.transposition[indexStep]
            transposition.backward(withinBatch, this.backwardWeightedAttendedEncodingWrtTransposedAttentionDistribution)
            val backwardTransposedAttentionDistributionWrtAttentionDistribution = transposition.backwardResult

            // d softmax(pre-activation) / d pre-activation
            val softmax = this.softmax[indexStep]
            softmax.backward(withinBatch, backwardTransposedAttentionDistributionWrtAttentionDistribution)
            val backwardAttentionDistributionWrtAttentionScores = softmax.backwardResult

            // d s * tanh(...) / d tanh (...)
            this.scoringWeighting.backwardStep(withinBatch, indexStep, backwardAttentionDistributionWrtAttentionScores)
            val backwardAttentionScoresWrtAttentionActivation = this.scoringWeighting.backwardResult

            // d tanh(...) / d W^e * E + expand(...)
            val tanh = this.tanh[indexStep]
            tanh.backward(withinBatch, backwardAttentionScoresWrtAttentionActivation)
            val backwardAttentionActivationWrtAttentionPreactivation = tanh.backwardResult

            // d W^e * E + expand(...) / d E
            this.encodingWeighting.backward(withinBatch, backwardAttentionActivationWrtAttentionPreactivation)
            val backwardAttentionPreactivationWrtEncodings = this.encodingWeighting.backwardResult
            this.inputAccumulator.accumulate(backwardAttentionPreactivationWrtEncodings)

            // d W^e * E + expand(W^d * d_t-1) / d W^d * d_t-1
            val columnRepetition = this.columnRepetitions[indexStep]
            columnRepetition.backward(withinBatch, backwardAttentionActivationWrtAttentionPreactivation)
            val backwardAttentionPreactivationWrtExpansion = columnRepetition.backwardResult

            //  d W^d * d_t-1 / d d_t-1
            this.attentionPreviousStateWeighting.backwardStep(withinBatch, indexStep, backwardAttentionPreactivationWrtExpansion)
            val backwardExpansionWrtWeightedPreviousState = this.attentionPreviousStateWeighting.backwardResult

            backwardNextDecoderStateWrtDecoderState = backwardPreActivationWrtWeightedPreviousStateForDecoding + backwardExpansionWrtWeightedPreviousState

        }

        // W^e is used once per series.

        // W^d
        this.attentionPreviousStateWeighting.backwardSeries()
        // s
        this.scoringWeighting.backwardSeries()
        // U^e
        this.attendedEncodingWeighting.backwardSeries()
        // U^d
        this.decodingPreviousDecoderWeighting.backwardSeries()
        // b
        this.bias?.backwardSeries()

        val encodingAccumulation = this.inputAccumulator.getAccumulation()
        System.arraycopy(encodingAccumulation, 0, this.backwardResult, 0, this.numberBackwardEntries)

        this.inputAccumulator.reset()

        return this.backwardResult

    }

    override fun optimize(batchSize : Int) {

        // W^e
        this.encodingWeighting.optimize(batchSize)
        // W^d
        this.attentionPreviousStateWeighting.optimize(batchSize)
        // s
        this.scoringWeighting.optimize(batchSize)
        // U^e
        this.attendedEncodingWeighting.optimize(batchSize)
        // U^d
        this.decodingPreviousDecoderWeighting.optimize(batchSize)
        // b
        this.bias?.optimize(batchSize)

    }

}