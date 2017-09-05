package com.komputation.loss

import com.komputation.cpu.loss.CpuSquaredLoss
import com.komputation.cuda.CudaContext
import com.komputation.cuda.kernels.LossKernels
import com.komputation.cuda.loss.CudaSquaredLoss

class SquaredLoss(private val numberRows: Int, private val numberColumns: Int) : CpuLossFunctionInstruction, CudaLossFunctionInstruction {

    override fun buildForCpu() =

        CpuSquaredLoss(this.numberRows, this.numberColumns)

    override fun buildForCuda(context: CudaContext): CudaSquaredLoss {

        return CudaSquaredLoss(
            this.numberRows,
            this.numberColumns,
            { context.createKernel(LossKernels.squaredLoss()) },
            { context.createKernel(LossKernels.backwardSquaredLoss()) },
            context.numberMultiprocessors,
            context.maximumNumberOfResidentWarpsPerMultiprocessor,
            context.warpSize,
            context.maximumNumberOfThreadsPerBlock)

    }

}

fun squaredLoss(numberCategories: Int, length: Int = 1) =

    SquaredLoss(numberCategories, length)