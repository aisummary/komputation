package com.komputation.layers.forward.activation

import jcuda.jcublas.cublasHandle
import com.komputation.cpu.layers.forward.activation.CpuExponentiationLayer
import com.komputation.cuda.CudaContext
import com.komputation.cuda.kernels.ForwardKernels
import com.komputation.cuda.layers.forward.activation.CudaExponentiationLayer
import com.komputation.layers.CpuForwardLayerInstruction
import com.komputation.layers.CudaForwardLayerInstruction

class ExponentiationLayer internal constructor(
    private val name : String?,
    private val numberRows: Int,
    private val numberColumns: Int,
    private val hasFixedLength: Boolean) : CpuForwardLayerInstruction, CudaForwardLayerInstruction {

    override fun buildForCpu() =

        CpuExponentiationLayer(this.name, this.numberRows, if(this.hasFixedLength) this.numberColumns else 1, this.numberColumns)

    override fun buildForCuda(context: CudaContext, cublasHandle: cublasHandle): CudaExponentiationLayer {

        val exponentiationLayer = CudaExponentiationLayer(
            this.name,
            this.numberRows,
            this.numberColumns,
            { context.createKernel(ForwardKernels.exponentiation()) },
            { context.createKernel(ForwardKernels.backwardExponentiation()) },
            context.warpSize,
            context.maximumNumberOfThreadsPerBlock)

        return exponentiationLayer

    }

}

fun exponentiationLayer(
    numberRows : Int,
    numberColumns: Int = 1,
    hasFixedLength: Boolean = true) =

    exponentiationLayer(
        null,
        numberRows,
        numberColumns,
        hasFixedLength
    )

fun exponentiationLayer(
    name : String?,
    numberRows : Int,
    numberColumns: Int = 1,
    hasFixedLength: Boolean = true) =

    ExponentiationLayer(
        name,
        numberRows,
        numberColumns,
        hasFixedLength)