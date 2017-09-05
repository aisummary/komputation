package com.komputation.optimization.historical

import com.komputation.cpu.optimization.CpuOptimizationStrategy
import com.komputation.cpu.optimization.historical.CpuNesterov
import com.komputation.cuda.CudaContext
import com.komputation.cuda.kernels.OptimizationKernels
import com.komputation.cuda.optimization.CudaOptimizationStrategy
import com.komputation.cuda.optimization.history.CudaNesterov
import com.komputation.optimization.OptimizationInstruction

fun nesterov(learningRate: Float, momentum : Float) =

    Nesterov(learningRate, momentum)

class Nesterov(private val learningRate: Float, private val momentum : Float) : OptimizationInstruction {

    override fun buildForCpu() : CpuOptimizationStrategy {

        return { numberRows : Int, numberColumns : Int ->

            CpuNesterov(this.learningRate, this.momentum, numberRows * numberColumns)

        }

    }

    override fun buildForCuda(context: CudaContext): CudaOptimizationStrategy {

        return { numberParameters : Int, numberRows : Int, numberColumns : Int ->

            CudaNesterov(
                numberParameters,
                numberRows * numberColumns,
                this.learningRate,
                this.momentum,
                { context.createKernel(OptimizationKernels.nesterov()) },
                context.numberMultiprocessors,
                context.maximumNumberOfResidentWarpsPerMultiprocessor,
                context.warpSize,
                context.maximumNumberOfThreadsPerBlock)

        }

    }

}