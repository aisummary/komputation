package com.komputation.optimization.adaptive

import com.komputation.cpu.optimization.CpuOptimizationStrategy
import com.komputation.cpu.optimization.adaptive.CpuAdadelta
import com.komputation.cuda.CudaContext
import com.komputation.cuda.optimization.CudaOptimizationStrategy
import com.komputation.optimization.OptimizationInstruction

fun adadelta(decay : Float = 0.95f, epsilon: Float = 1e-6f) =

    Adadelta(decay, epsilon)

class Adadelta(private val decay : Float, private val epsilon: Float) : OptimizationInstruction {

    override fun buildForCpu() : CpuOptimizationStrategy {

        return { numberRows : Int, numberColumns : Int ->

            CpuAdadelta(this.decay, this.epsilon, numberRows * numberColumns)

        }

    }

    override fun buildForCuda(context: CudaContext): CudaOptimizationStrategy {

        throw NotImplementedError()

    }

}