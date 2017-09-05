package com.komputation.layers.forward.activation

import jcuda.jcublas.cublasHandle
import com.komputation.cpu.layers.forward.activation.CpuTanhLayer
import com.komputation.cuda.CudaContext
import com.komputation.cuda.kernels.ForwardKernels
import com.komputation.cuda.layers.forward.activation.CudaTanhLayer
import com.komputation.layers.CpuActivationLayerInstruction
import com.komputation.layers.CudaActivationLayerInstruction

class TanhLayer internal constructor(
    private val name : String?,
    private val numberRows : Int,
    private val numberColumns : Int,
    private val hasFixedLength: Boolean) : CpuActivationLayerInstruction, CudaActivationLayerInstruction {

    override fun buildForCpu() =

        CpuTanhLayer(this.name, this.numberRows, if(this.hasFixedLength) this.numberColumns else 1, this.numberColumns)

    override fun buildForCuda(context: CudaContext, cublasHandle: cublasHandle): CudaTanhLayer {

        return CudaTanhLayer(
            this.name,
            this.numberRows,
            this.numberColumns,
            { context.createKernel(ForwardKernels.tanh()) },
            { context.createKernel(ForwardKernels.backwardTanh()) },
            context.warpSize,
            context.maximumNumberOfThreadsPerBlock)

    }

}

fun tanhLayer(numberRows : Int, numberColumns: Int = 1, isFixedLength: Boolean = true) =

    tanhLayer(null, numberRows, numberColumns, isFixedLength)

fun tanhLayer(name : String? = null, numberRows : Int, numberColumns: Int = 1, isFixedLength: Boolean = true) =

    TanhLayer(name, numberRows, numberColumns, isFixedLength)