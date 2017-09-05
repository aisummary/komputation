package com.komputation.cpu.layers.forward

import com.komputation.cpu.functions.splitRows
import com.komputation.cpu.functions.stackRows
import com.komputation.cpu.layers.BaseCpuVariableLengthForwardLayer
import com.komputation.cpu.layers.CpuForwardLayer
import com.komputation.layers.Resourceful
import com.komputation.optimization.Optimizable

class CpuConcatenation internal constructor(
    name : String? = null,
    numberInputRows: Int,
    minimumColumns : Int,
    maximumColumns : Int,
    private val heights: IntArray,
    private val width : Int,
    private val layers: Array<CpuForwardLayer>) : BaseCpuVariableLengthForwardLayer(name, numberInputRows, heights.sum(), minimumColumns, maximumColumns), Resourceful, Optimizable {

    private val numberLayers = layers.size
    private val individualResults = Array(this.numberLayers) { FloatArray(0) }

    private var chainSplit = emptyArray<FloatArray>()

    override fun acquire(maximumBatchSize: Int) {

        super.acquire(maximumBatchSize)

        this.chainSplit = Array(this.numberLayers) { index -> FloatArray(this.heights[index]) }

    }

    override fun release() {

    }

    override fun computeNumberOutputColumns(lengthIndex: Int, length: Int) = this.width

    override fun computeForwardResult(withinBatch: Int, numberInputColumns: Int, input: FloatArray, isTraining: Boolean, result: FloatArray) {

        for (indexLayer in (0..this.numberLayers-1)) {

            this.individualResults[indexLayer] = this.layers[indexLayer].forward(withinBatch, numberInputColumns, input, isTraining)

        }

        stackRows(this.heights, this.numberOutputRows, this.width, result, *this.individualResults)

    }

    override fun computeBackwardResult(withinBatch: Int, chain: FloatArray, result: FloatArray) {

        splitRows(this.numberOutputRows, this.numberOutputColumns, chain, this.heights, this.numberLayers, this.chainSplit)

        val firstLayer = this.layers[0]
        firstLayer.backward(withinBatch, this.chainSplit[0])

        val firstIndividualBackwardResult = firstLayer.backwardResult

        System.arraycopy(firstIndividualBackwardResult, 0, result, 0, firstIndividualBackwardResult.size)

        for (indexNetwork in (1..this.numberLayers-1)) {

            val layer = this.layers[indexNetwork]

            layer.backward(withinBatch, this.chainSplit[indexNetwork])

            val individualBackwardResult = layer.backwardResult

            for (index in 0..individualBackwardResult.size - 1) {

                result[index] += individualBackwardResult[index]

            }

        }

    }

    override fun optimize(batchSize : Int) {

        for (layer in this.layers) {

            if (layer is Optimizable) {

                layer.optimize(batchSize)

            }

        }

    }

}