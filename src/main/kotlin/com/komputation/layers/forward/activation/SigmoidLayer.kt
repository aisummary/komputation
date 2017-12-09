package com.komputation.layers.forward.activation

import com.komputation.cpu.layers.forward.activation.CpuSigmoidLayer
import com.komputation.cuda.CudaContext
import com.komputation.cuda.kernels.ForwardKernels
import com.komputation.cuda.layers.forward.activation.CudaSigmoidLayer
import jcuda.jcublas.cublasHandle

class SigmoidLayer internal constructor(
    private val name : String?,
    private val numberRows: Int,
    private val numberColumns: Int,
    private val hasFixedLength: Boolean) : ActivationLayerInstruction {

    override fun buildForCpu() =
        CpuSigmoidLayer(this.name, this.numberRows, if(this.hasFixedLength) this.numberColumns else 1, this.numberColumns)

    override fun buildForCuda(context : CudaContext, cublasHandle: cublasHandle) =
        CudaSigmoidLayer(
            this.name,
            this.numberRows,
            this.numberColumns,
            { context.createKernel(ForwardKernels.sigmoid()) },
            { context.createKernel(ForwardKernels.backwardSigmoid()) },
            context.warpSize,
            context.maximumNumberOfThreadsPerBlock)

}

fun sigmoidLayer(numberRows : Int, numberColumns: Int = 1, hasFixedLength: Boolean = true) =
    sigmoidLayer(null, numberRows, numberColumns, hasFixedLength)

fun sigmoidLayer(name : String? = null, numberRows : Int, numberColumns: Int = 1, hasFixedLength: Boolean = true) =
    SigmoidLayer(name, numberRows, numberColumns, hasFixedLength)