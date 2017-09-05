package com.komputation.optimization.adaptive

import com.komputation.cpu.optimization.CpuOptimizationStrategy
import com.komputation.cpu.optimization.adaptive.CpuAdagrad
import com.komputation.cuda.CudaContext
import com.komputation.cuda.optimization.CudaOptimizationStrategy
import com.komputation.optimization.OptimizationInstruction

fun adagrad(learningRate: Float, epsilon: Float = 1e-6f) =

    Adagrad(learningRate, epsilon)

class Adagrad(private val learningRate: Float, private val epsilon: Float) : OptimizationInstruction {

    override fun buildForCpu() : CpuOptimizationStrategy {

        return { numberRows : Int, numberColumns : Int ->

            CpuAdagrad(this.learningRate, this.epsilon, numberRows * numberColumns)

        }

    }

    override fun buildForCuda(context: CudaContext): CudaOptimizationStrategy {

        throw NotImplementedError()

    }

}