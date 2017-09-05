package com.komputation.optimization

import com.komputation.cpu.optimization.CpuOptimizationStrategy
import com.komputation.cpu.optimization.CpuStochasticGradientDescent
import com.komputation.cuda.CudaContext
import com.komputation.cuda.kernels.OptimizationKernels
import com.komputation.cuda.optimization.CudaOptimizationStrategy
import com.komputation.cuda.optimization.CudaStochasticGradientDescent

fun stochasticGradientDescent(learningRate: Float) =

    StochasticGradientDescent(learningRate)

class StochasticGradientDescent(private val learningRate: Float) : OptimizationInstruction {

    override fun buildForCpu() : CpuOptimizationStrategy {

        return { _: Int, _: Int ->

            CpuStochasticGradientDescent(this.learningRate)

        }

    }

    override fun buildForCuda(context: CudaContext): CudaOptimizationStrategy {

        return { _ : Int, numberRows : Int, numberColumns : Int ->

            CudaStochasticGradientDescent(
                numberRows * numberColumns,
                this.learningRate,
                { context.createKernel(OptimizationKernels.stochasticGradientDescent()) },
                context.numberMultiprocessors,
                context.maximumNumberOfResidentWarpsPerMultiprocessor,
                context.warpSize,
                context.maximumNumberOfThreadsPerBlock)

        }

    }

}