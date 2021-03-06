package com.komputation.cuda.workflow

import com.komputation.cuda.loss.CudaLossFunction
import com.komputation.cuda.memory.InputMemory
import com.komputation.cuda.memory.TargetMemory
import com.komputation.cuda.network.CudaBackwardPropagator
import com.komputation.cuda.network.CudaForwardPropagator
import com.komputation.matrix.Matrix
import com.komputation.matrix.partitionIndices
import com.komputation.optimization.Optimizable
import jcuda.Pointer

class CudaTrainer(
    private val forwardPropagator : CudaForwardPropagator,
    private val backwardPropagator : CudaBackwardPropagator,
    private val optimizables : Array<Optimizable>,
    private val inputs : Array<out Matrix>,
    private val targets: Array<FloatArray>,
    private val numberIterations : Int,
    private val maximumBatchSize : Int,
    private val lossFunction : CudaLossFunction,
    private val afterEachIteration : ((index : Int, loss : Float) -> Unit)? = null) {

    private val numberExamples = this.inputs.size

    private val batches = partitionIndices(this.numberExamples, this.maximumBatchSize)

    private val inputMemory = InputMemory()
    private val targetMemory = TargetMemory(this.targets.first().size)

    init {
        this.lossFunction.acquire(this.maximumBatchSize)
    }

    fun free() {
        this.lossFunction.release()

        this.inputMemory.free()
        this.targetMemory.free()
    }

    fun run(): Pair<Long, Pair<List<Pair<String?, Long>>, List<Pair<String?, Long>>>> {
        val trackLoss = this.afterEachIteration != null

        val start = System.currentTimeMillis()

        repeat(this.numberIterations) { indexIteration ->
            var iterationLoss = if(trackLoss) 0f else Float.NaN

            for ((batchId, batch) in this.batches.withIndex()) {
                val currentBatchSize = batch.size

                val devicePredictions = this.forwardPropagator.forward(batchId, currentBatchSize, batch, this.inputs, this.inputMemory,true)
                val pointerToPredictions = Pointer.to(devicePredictions)

                val pointerToTargets = this.targetMemory.get(batchId, currentBatchSize, batch, this.targets)

                if (trackLoss) {
                    this.lossFunction.accumulate(pointerToPredictions, pointerToTargets, currentBatchSize)
                }

                val backwardLoss = this.lossFunction.backward(currentBatchSize, pointerToPredictions, pointerToTargets)

                this.backwardPropagator.backward(batchId, currentBatchSize, backwardLoss, this.inputMemory)

                for (optimizable in this.optimizables) {
                    optimizable.optimize(currentBatchSize)
                }

                if (trackLoss) {
                    iterationLoss += this.lossFunction.accessAccumulation()
                }
            }

            this.afterEachIteration?.invoke(indexIteration, iterationLoss)
        }

        val forwardPropagationTimes = this.forwardPropagator.stopTimer()
        val backwardPropagationTimes = this.backwardPropagator.stopTimer()

        val stop = System.currentTimeMillis()

        val totalTime = stop - start

        return totalTime to (forwardPropagationTimes to backwardPropagationTimes)
    }

}