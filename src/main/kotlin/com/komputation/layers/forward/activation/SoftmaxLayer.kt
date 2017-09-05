package com.komputation.layers.forward.activation

import jcuda.jcublas.cublasHandle
import com.komputation.cpu.layers.forward.activation.CpuSoftmaxLayer
import com.komputation.cuda.CudaContext
import com.komputation.cuda.layers.forward.activation.CudaSoftmaxLayer
import com.komputation.layers.CpuActivationLayerInstruction
import com.komputation.layers.CudaActivationLayerInstruction
import com.komputation.layers.concatenateNames
import com.komputation.layers.forward.normalization.normalizationLayer

class SoftmaxLayer internal constructor(
    private val name : String?,
    private val numberRows : Int,
    private val numberColumns : Int,
    private val hasFixedLength: Boolean) : CpuActivationLayerInstruction, CudaActivationLayerInstruction {

    private val exponentiationLayer = exponentiationLayer(concatenateNames(this.name, "exponentiation"), this.numberRows, this.numberColumns, this.hasFixedLength)
    private val normalizationLayer = normalizationLayer(concatenateNames(this.name, "normalization"), this.numberRows, this.numberColumns, this.hasFixedLength)

    override fun buildForCpu() =

        CpuSoftmaxLayer(this.name, this.exponentiationLayer.buildForCpu(), this.normalizationLayer.buildForCpu())

    override fun buildForCuda(context: CudaContext, cublasHandle: cublasHandle) =

        CudaSoftmaxLayer(this.name, this.exponentiationLayer.buildForCuda(context, cublasHandle), this.normalizationLayer.buildForCuda(context, cublasHandle))

}

fun softmaxLayer(numberCategories: Int, numberSteps: Int = 1, hasFixedLength: Boolean = true) =

    softmaxLayer(null, numberCategories, numberSteps, hasFixedLength)

fun softmaxLayer(name : String? = null, numberCategories: Int, numberSteps: Int = 1, hasFixedLength: Boolean = true) =

    SoftmaxLayer(name, numberCategories, numberSteps, hasFixedLength)