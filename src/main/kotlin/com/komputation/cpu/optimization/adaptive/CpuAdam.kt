package com.komputation.cpu.optimization.adaptive

import com.komputation.cpu.optimization.UpdateRule
import com.komputation.matrix.FloatMath

class CpuAdam(
    private val learningRate : Float,
    private val firstMomentDecay : Float,
    private val secondMomentDecay : Float,
    private val epsilon : Float,
    size : Int) : UpdateRule {

    private val oneMinusFirstMomentDecay = 1.0f - this.firstMomentDecay
    private val oneMinusSecondMomentDecay = 1.0f - this.secondMomentDecay

    private val firstMomentEstimate = FloatArray(size)
    private val secondMomentEstimate = FloatArray(size)

    private var step = 0.0f

    override fun updateSparsely(start : Int, parameters: FloatArray, gradient: FloatArray, numberEntries: Int) {

        this.step += 1.0f

        for (index in 0..numberEntries - 1) {

            val derivative = gradient[index]

            val updatedFirstMomentEstimate = this.firstMomentDecay * this.firstMomentEstimate[index] + this.oneMinusFirstMomentDecay * derivative
            this.firstMomentEstimate[index] = updatedFirstMomentEstimate
            val correctedFirstMomentEstimate = updatedFirstMomentEstimate / (1.0f - FloatMath.pow(this.firstMomentDecay, this.step))

            val updatedSecondMomentEstimate = this.secondMomentDecay * this.secondMomentEstimate[index] + this.oneMinusSecondMomentDecay * derivative * derivative
            this.secondMomentEstimate[index] = updatedSecondMomentEstimate
            val correctedSecondMomentEstimate = updatedSecondMomentEstimate / (1.0f - FloatMath.pow(this.secondMomentDecay, this.step))

            val adaptedLearningRate = this.learningRate / (FloatMath.sqrt(correctedSecondMomentEstimate) + this.epsilon)

            val change = -correctedFirstMomentEstimate * adaptedLearningRate

            parameters[index] += change

        }

    }

}