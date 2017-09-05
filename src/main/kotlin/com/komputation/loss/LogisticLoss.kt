package com.komputation.loss

import com.komputation.cpu.loss.CpuLogisticLoss
import com.komputation.cuda.CudaContext
import com.komputation.cuda.kernels.LossKernels
import com.komputation.cuda.loss.CudaLogisticLoss

class LogisticLoss(private val numberRows: Int, private val numberColumns: Int) : CpuLossFunctionInstruction, CudaLossFunctionInstruction {

    override fun buildForCpu() =

        CpuLogisticLoss(this.numberRows, this.numberColumns)

    override fun buildForCuda(context: CudaContext): CudaLogisticLoss {

        return CudaLogisticLoss(
            this.numberRows,
            this.numberColumns,
            { blockSize : Int -> context.createKernel(LossKernels.logisticLoss(blockSize)) },
            { context.createKernel(LossKernels.backwardLogisticLoss()) },
            context.numberMultiprocessors,
            context.maximumNumberOfResidentWarpsPerMultiprocessor,
            context.warpSize,
            context.maximumNumberOfThreadsPerBlock)

    }

}

fun logisticLoss(numberCategories: Int, length: Int = 1) =

    LogisticLoss(numberCategories, length)