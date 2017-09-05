package com.komputation.optimization.adaptive

import com.komputation.cpu.optimization.CpuOptimizationStrategy
import com.komputation.cpu.optimization.adaptive.CpuAdam
import com.komputation.cuda.CudaContext
import com.komputation.cuda.optimization.CudaOptimizationStrategy
import com.komputation.optimization.OptimizationInstruction

fun adam(learningRate : Float = 0.001f, firstMomentDecay : Float = 0.9f, secondMomentDecay : Float = 0.999f, epsilon : Float = 1e-8f) =

    Adam(learningRate, firstMomentDecay, secondMomentDecay, epsilon)

class Adam(
    private val learningRate : Float,
    private val firstMomentDecay : Float,
    private val secondMomentDecay : Float,
    private val epsilon : Float) : OptimizationInstruction {

    override fun buildForCpu() : CpuOptimizationStrategy {

        return { numberRows : Int, numberColumns : Int ->

            CpuAdam(this.learningRate, this.firstMomentDecay, this.secondMomentDecay, this.epsilon, numberRows * numberColumns)

        }

    }

    override fun buildForCuda(context: CudaContext): CudaOptimizationStrategy {

        throw NotImplementedError()

    }

}