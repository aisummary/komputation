package com.komputation.layers.forward.dropout

import jcuda.jcublas.cublasHandle
import com.komputation.cpu.layers.forward.dropout.CpuDropoutLayer
import com.komputation.cuda.CudaContext
import com.komputation.cuda.kernels.ForwardKernels
import com.komputation.cuda.layers.forward.dropout.CudaDropoutLayer
import com.komputation.layers.CpuForwardLayerInstruction
import com.komputation.layers.CudaForwardLayerInstruction
import java.util.*

class DropoutLayer internal constructor(
    private val name : String?,
    private val numberRows : Int,
    private val numberColumns : Int,
    private val random : Random,
    private val keepProbability : Float) : CpuForwardLayerInstruction, CudaForwardLayerInstruction {

    override fun buildForCpu() =

        CpuDropoutLayer(this.name, this.numberRows, this.numberColumns, this.random, this.keepProbability)

    override fun buildForCuda(context: CudaContext, cublasHandle: cublasHandle): CudaDropoutLayer {

        return CudaDropoutLayer(
            this.name,
            this.numberRows,
            this.numberColumns,
            this.random,
            this.keepProbability,
            { context.createKernel(ForwardKernels.dropoutTraining()) },
            { context.createKernel(ForwardKernels.dropoutRuntime()) },
            { context.createKernel(ForwardKernels.backwardDropout()) },
            context.warpSize,
            context.maximumNumberOfThreadsPerBlock)

    }

}

fun dropoutLayer(random: Random, keepProbability: Float, numberRows: Int, numberColumns: Int = 1) =

    dropoutLayer(null, numberColumns, keepProbability, random, numberRows)

fun dropoutLayer(name: String?, numberColumns: Int, keepProbability: Float, random: Random, numberRows: Int) =

    DropoutLayer(name, numberRows, numberColumns, random, keepProbability)