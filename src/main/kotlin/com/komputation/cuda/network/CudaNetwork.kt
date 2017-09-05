package com.komputation.cuda.network

import jcuda.jcublas.JCublas2.cublasCreate
import jcuda.jcublas.JCublas2.cublasDestroy
import jcuda.jcublas.cublasHandle
import com.komputation.cuda.CudaEvaluation
import com.komputation.cuda.kernels.EvaluationKernels
import com.komputation.cuda.setUpCudaContext
import com.komputation.cuda.workflow.CudaTester
import com.komputation.cuda.workflow.CudaTrainer
import com.komputation.layers.CudaEntryPointInstruction
import com.komputation.layers.CudaForwardLayerInstruction
import com.komputation.layers.Resourceful
import com.komputation.layers.acquireRecursively
import com.komputation.loss.CudaLossFunctionInstruction
import com.komputation.matrix.Matrix
import com.komputation.optimization.Optimizable


class CudaNetwork(
    private val maximumBatchSize: Int,
    entryPointInstruction: CudaEntryPointInstruction,
    vararg forwardLayerInstructions: CudaForwardLayerInstruction) {

    private val cudaContext = setUpCudaContext()
    private val cublasHandle = cublasHandle()

    private val entryPoint = entryPointInstruction.buildForCuda(this.cudaContext)
    private val hasFixedLengthInput = this.entryPoint.hasFixedLength

    private val layers = Array(forwardLayerInstructions.size) { index -> forwardLayerInstructions[index].buildForCuda(this.cudaContext, this.cublasHandle) }
    private val optimizables = listOf(this.entryPoint).plus(this.layers).filterIsInstance(Optimizable::class.java).reversed().toTypedArray()

    private val forwardPropagator =

        if(this.hasFixedLengthInput)
            CudaFixedLengthForwardPropagator(this.entryPoint, this.layers)
        else
            CudaVariableLengthForwardPropagator(this.entryPoint, this.layers)

    private val backwardPropagator =

        CudaBackwardPropagator(this.entryPoint, this.layers)

    init {

        cublasCreate(this.cublasHandle)

        acquireRecursively(this.entryPoint, this.maximumBatchSize)

        for (layer in this.layers) {

            acquireRecursively(layer, this.maximumBatchSize)

        }

    }

    fun free() {

        cublasDestroy(this.cublasHandle)

        for (layer in this.layers) {

            if (layer is Resourceful) {

                layer.release()

            }

        }

        if (this.entryPoint is Resourceful) {

            this.entryPoint.release()

        }

    }

    fun training(
        inputs: Array<Matrix>,
        targets: Array<FloatArray>,
        numberIterations : Int,
        lossFunction : CudaLossFunctionInstruction,
        afterEachIteration : ((index : Int, loss : Float) -> Unit)? = null) =

        CudaTrainer(
            this.forwardPropagator,
            this.backwardPropagator,
            this.optimizables,
            inputs,
            targets,
            numberIterations,
            this.maximumBatchSize,
            lossFunction.buildForCuda(this.cudaContext),
            afterEachIteration
        )


    fun test(
        inputs: Array<Matrix>,
        targets: Array<FloatArray>,
        batchSize: Int,
        numberCategories : Int,
        length : Int = 1) =

        CudaTester(
            this.forwardPropagator,
            CudaEvaluation(inputs.size, numberCategories, length, { this.cudaContext.createKernel(EvaluationKernels.evaluation()) }),
            inputs,
            targets,
            batchSize
        )

}