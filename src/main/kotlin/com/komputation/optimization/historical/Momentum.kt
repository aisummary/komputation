package com.komputation.optimization.historical

import com.komputation.cpu.optimization.CpuOptimizationStrategy
import com.komputation.cpu.optimization.historical.CpuMomentum
import com.komputation.cuda.CudaContext
import com.komputation.cuda.kernels.OptimizationKernels
import com.komputation.cuda.optimization.CudaOptimizationStrategy
import com.komputation.cuda.optimization.history.CudaMomentum
import com.komputation.optimization.OptimizationInstruction

fun momentum(learningRate: Float, momentum : Float) =

    Momentum(learningRate, momentum)

class Momentum(private val learningRate: Float, private val momentum : Float) : OptimizationInstruction {

    override fun buildForCpu() : CpuOptimizationStrategy {

        return { numberRows : Int, numberColumns : Int ->

            CpuMomentum(this.learningRate, this.momentum, numberRows * numberColumns)

        }

    }

    override fun buildForCuda(context: CudaContext): CudaOptimizationStrategy {

        return { numberParameters : Int, numberRows : Int, numberColumns : Int ->

            CudaMomentum(
                numberParameters,
                numberRows * numberColumns,
                this.learningRate,
                this.momentum,
                { context.createKernel(OptimizationKernels.momentum()) },
                context.numberMultiprocessors,
                context.maximumNumberOfResidentWarpsPerMultiprocessor,
                context.warpSize,
                context.maximumNumberOfThreadsPerBlock)

        }

    }

}