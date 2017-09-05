package com.komputation.layers.forward.normalization

import jcuda.jcublas.cublasHandle
import com.komputation.cpu.layers.forward.normalization.CpuNormalizationLayer
import com.komputation.cuda.CudaContext
import com.komputation.cuda.kernels.ForwardKernels
import com.komputation.cuda.layers.forward.normalization.CudaNormalizationLayer
import com.komputation.layers.CpuForwardLayerInstruction
import com.komputation.layers.CudaForwardLayerInstruction

class NormalizationLayer internal constructor(
    private val name : String?,
    private val numberRows : Int,
    private val numberColumns : Int,
    private val isFixedLength: Boolean) : CpuForwardLayerInstruction, CudaForwardLayerInstruction {

    override fun buildForCpu() =

        CpuNormalizationLayer(this.name, this.numberRows, if (this.isFixedLength) this.numberColumns else 1, this.numberColumns)

    override fun buildForCuda(context: CudaContext, cublasHandle: cublasHandle): CudaNormalizationLayer {

        val normalizationLayer = CudaNormalizationLayer(
            this.name,
            this.numberRows,
            this.numberColumns,
            { context.createKernel(ForwardKernels.normalization()) },
            { context.createKernel(ForwardKernels.backwardNormalization()) },
            context.maximumNumberOfThreadsPerBlock,
            context.warpSize)

        return normalizationLayer

    }

}

fun normalizationLayer(
    numberRows : Int,
    numberColumns: Int = 1,
    isFixedLength : Boolean = true) =

    normalizationLayer(null, numberRows, numberColumns, isFixedLength)

fun normalizationLayer(
    name : String?,
    numberRows : Int,
    numberColumns: Int = 1,
    isFixedLength : Boolean = true) =

    NormalizationLayer(name, numberRows, numberColumns, isFixedLength)