package com.komputation.layers.forward.activation

import jcuda.jcublas.cublasHandle
import com.komputation.cpu.layers.forward.activation.CpuIdentityLayer
import com.komputation.cuda.CudaContext
import com.komputation.cuda.layers.forward.activation.CudaIdentityLayer
import com.komputation.layers.CpuActivationLayerInstruction
import com.komputation.layers.CudaActivationLayerInstruction

class IdentityLayer internal constructor(
    private val name : String?,
    private val numberRows : Int,
    private val numberColumns : Int) : CpuActivationLayerInstruction, CudaActivationLayerInstruction {

    override fun buildForCpu() =

        CpuIdentityLayer(this.name, this.numberRows, this.numberColumns)

    override fun buildForCuda(context: CudaContext, cublasHandle: cublasHandle) =

        CudaIdentityLayer(this.name, this.numberRows, this.numberColumns)
}

fun identityLayer(name : String? = null, numberRows : Int, numberColumns : Int = 1) =

    IdentityLayer(name, numberRows, numberColumns)