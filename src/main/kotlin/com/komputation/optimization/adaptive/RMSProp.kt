package com.komputation.optimization.adaptive

import com.komputation.cpu.optimization.CpuOptimizationStrategy
import com.komputation.cpu.optimization.adaptive.CpuRMSProp
import com.komputation.cuda.CudaContext
import com.komputation.cuda.optimization.CudaOptimizationStrategy
import com.komputation.optimization.OptimizationInstruction

fun rmsprop(learningRate: Float, decay : Float = 0.9f, epsilon: Float = 1e-6f) =

    RMSProp(learningRate, decay, epsilon)

class RMSProp(private val learningRate: Float, private val decay : Float, private val epsilon: Float) : OptimizationInstruction {

    override fun buildForCpu() : CpuOptimizationStrategy {

        return { numberRows : Int, numberColumns : Int ->

            CpuRMSProp(this.learningRate, this.decay, this.epsilon, numberRows * numberColumns)

        }

    }

    override fun buildForCuda(context: CudaContext): CudaOptimizationStrategy {

        throw NotImplementedError()

    }

}