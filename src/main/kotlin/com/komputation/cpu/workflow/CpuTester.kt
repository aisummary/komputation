package com.komputation.cpu.workflow

import com.komputation.cpu.network.CpuForwardPropagator
import com.komputation.cpu.evaluation.computeAccuracy
import com.komputation.cpu.functions.findMaxIndices
import com.komputation.matrix.Matrix
import com.komputation.matrix.partitionIndices

class CpuTester(
    private val forwardPropagator: CpuForwardPropagator,
    private val inputs : Array<Matrix>,
    private val targets: Array<FloatArray>,
    batchSize : Int,
    private val numberCategories : Int,
    private val length : Int) {

    private val numberInstances = inputs.size
    private val batches = partitionIndices(inputs.size, batchSize)

    private val actualCategories = Array(this.numberInstances) { IntArray(this.length) }
    private val predictedCategories = Array(this.numberInstances) { IntArray(this.length) }

    fun run(): Float {

        for((withinBatch, batch) in batches.withIndex()) {

            for (index in batch) {

                val input = this.inputs[index]

                val batchTargets = this.targets[index]
                findMaxIndices(batchTargets, this.numberCategories, this.length, this.actualCategories[index])

                val batchPredictions = this.forwardPropagator.forward(withinBatch, input, false)
                findMaxIndices(batchPredictions, this.numberCategories, this.length, this.predictedCategories[index])

            }

        }

        val accuracy = computeAccuracy(this.actualCategories, this.predictedCategories, this.numberInstances)

        return accuracy

    }

}