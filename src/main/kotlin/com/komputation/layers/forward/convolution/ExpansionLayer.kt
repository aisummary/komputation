package com.komputation.layers.forward.convolution

import jcuda.jcublas.cublasHandle
import com.komputation.cpu.layers.forward.convolution.CpuExpansionLayer
import com.komputation.cuda.CudaContext
import com.komputation.cuda.kernels.ForwardKernels
import com.komputation.cuda.layers.forward.convolution.CudaExpansionLayer
import com.komputation.layers.CpuForwardLayerInstruction
import com.komputation.layers.CudaForwardLayerInstruction


class ExpansionLayer internal constructor(
    private val name : String?,
    private val numberInputRows : Int,
    private val maximumInputColumns: Int,
    private val hasFixedLength: Boolean,
    private val numberFilterRowPositions: Int,
    private val filterWidth: Int,
    private val filterHeight: Int) : CpuForwardLayerInstruction, CudaForwardLayerInstruction {

    private val minimumInputLength = if(this.hasFixedLength) this.maximumInputColumns else this.filterWidth

    private val filterLength = this.filterWidth * this.filterHeight

    override fun buildForCpu() =

        CpuExpansionLayer(
            this.name,
            this.numberInputRows,
            this.minimumInputLength,
            this.maximumInputColumns,
            this.numberFilterRowPositions,
            this.filterLength,
            this.filterWidth,
            this.filterHeight)

    override fun buildForCuda(context: CudaContext, cublasHandle: cublasHandle) =

        CudaExpansionLayer(
            this.name,
            this.numberInputRows,
            this.maximumInputColumns,
            this.filterHeight,
            this.filterWidth,
            { context.createKernel(ForwardKernels.expansion()) },
            { context.createKernel(ForwardKernels.backwardExpansion()) },
            context.warpSize,
            context.maximumNumberOfThreadsPerBlock)


}

fun expansionLayer(
    numberInputRows : Int,
    numberInputColumns : Int,
    hasFixedLength: Boolean,
    numberFilterRowPositions: Int,
    filterWidth: Int,
    filterHeight: Int) =

    expansionLayer(null, numberInputRows, numberInputColumns, hasFixedLength, numberFilterRowPositions, filterWidth, filterHeight)

fun expansionLayer(
    name : String?,
    numberInputRows : Int,
    numberInputColumns : Int,
    hasFixedLength: Boolean,
    numberFilterRowPositions: Int,
    filterWidth: Int,
    filterHeight: Int) =

    ExpansionLayer(name, numberInputRows, numberInputColumns, hasFixedLength, numberFilterRowPositions, filterWidth, filterHeight)