package com.komputation.cpu.layers.forward.projection

import com.komputation.cpu.functions.backwardProjectionWrtInput
import com.komputation.cpu.functions.backwardProjectionWrtWeights
import com.komputation.cpu.functions.multiply
import com.komputation.cpu.layers.BaseCpuForwardLayer
import com.komputation.cpu.optimization.DenseAccumulator
import com.komputation.cpu.optimization.UpdateRule
import com.komputation.cpu.optimization.updateDensely
import com.komputation.layers.Resourceful
import com.komputation.optimization.Optimizable

class CpuWeightingLayer internal constructor(
    name : String? = null,
    private val weights : FloatArray,
    override val numberInputRows : Int,
    private val minimumInputColumns: Int,
    private val maximumInputColumns: Int,
    override val numberOutputRows: Int,
    private val weightAccumulator : DenseAccumulator,
    private val weightUpdateRule: UpdateRule? = null) : BaseCpuForwardLayer(name), Resourceful, Optimizable {

    private val numberLengths = this.maximumInputColumns - this.minimumInputColumns + 1
    private val lengths = IntArray(this.numberLengths) { index -> index + this.minimumInputColumns }

    private val numberWeightColumns = this.numberInputRows
    private val numberWeightRows = this.numberOutputRows
    private val numberWeightEntries = this.numberWeightRows * this.numberWeightColumns

    override var numberInputColumns = -1
    private var blasInputMatricesOverPossibleLengths = emptyArray<org.jblas.FloatMatrix>()

    override var numberOutputColumns = -1
    private var blasOutputMatricesOverPossibleLengths = emptyArray<org.jblas.FloatMatrix>()
    override var forwardResult = FloatArray(0)

    private var backwardResultsOverPossibleLengths = emptyArray<FloatArray>()
    override var backwardResult = FloatArray(0)

    private val backwardWrtWeights = FloatArray(this.numberWeightEntries)

    private var blasWeightMatrix = org.jblas.FloatMatrix()

    private var lengthIndex = -1

    override fun acquire(maximumBatchSize: Int) {

        this.blasOutputMatricesOverPossibleLengths = Array(this.numberLengths) { index -> org.jblas.FloatMatrix(this.numberOutputRows, this.lengths[index]) }
        this.blasInputMatricesOverPossibleLengths = Array(this.numberLengths) { index -> org.jblas.FloatMatrix(this.numberInputRows, this.lengths[index]) }
        this.backwardResultsOverPossibleLengths = Array(this.numberLengths) { index -> FloatArray(this.numberInputRows * this.lengths[index]) }
        this.blasWeightMatrix = org.jblas.FloatMatrix(this.numberWeightRows, this.numberWeightColumns)
        this.blasWeightMatrix.data = this.weights

    }

    override fun release() {

    }

    override fun forward(withinBatch : Int, numberInputColumns: Int, input: FloatArray, isTraining : Boolean): FloatArray {

        this.numberInputColumns = numberInputColumns
        this.numberOutputColumns = numberInputColumns

        this.lengthIndex = numberInputColumns - this.minimumInputColumns

        val blasInputMatrix = this.blasInputMatricesOverPossibleLengths[this.lengthIndex]
        blasInputMatrix.data = input

        val blasOutputMatrix = this.blasOutputMatricesOverPossibleLengths[this.lengthIndex]

        multiply(this.blasWeightMatrix, blasInputMatrix, blasOutputMatrix)

        this.forwardResult = blasOutputMatrix.data

        return this.forwardResult

    }

    override fun backward(withinBatch : Int, chain : FloatArray): FloatArray {

        this.backwardResult = backwardResultsOverPossibleLengths[this.lengthIndex]
        backwardProjectionWrtInput(
            this.numberInputRows,
            this.numberInputColumns,
            this.weights,
            this.numberWeightRows,
            chain,
            this.numberOutputRows,
            this.backwardResult)

        val blasInputMatrix = this.blasInputMatricesOverPossibleLengths[this.lengthIndex]
        backwardProjectionWrtWeights(
            this.numberWeightRows,
            this.numberWeightColumns,
            blasInputMatrix.data,
            this.numberInputRows,
            chain,
            this.numberOutputRows,
            this.numberOutputColumns,
            this.backwardWrtWeights)

        this.weightAccumulator.accumulate(this.backwardWrtWeights)

        return this.backwardResult

    }

    override fun optimize(batchSize : Int) {

        if (this.weightUpdateRule != null) {

            updateDensely(this.weights, this.weightAccumulator.getAccumulation(), this.numberWeightEntries, batchSize, this.weightUpdateRule)

            this.weightAccumulator.reset()

        }

    }

}