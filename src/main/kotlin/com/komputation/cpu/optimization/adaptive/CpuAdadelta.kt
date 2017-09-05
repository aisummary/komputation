package com.komputation.cpu.optimization.adaptive

import com.komputation.cpu.optimization.UpdateRule
import com.komputation.matrix.FloatMath


class CpuAdadelta(private val decay : Float, private val epsilon : Float, size : Int) : UpdateRule {

    private val oneMinusDecay = 1.0f - this.decay

    private val gradientAccumulation = FloatArray(size)
    private val updateAccumulation = FloatArray(size)

    override fun updateSparsely(start : Int, parameters: FloatArray, gradient: FloatArray, numberEntries: Int) {

        for(localIndex in 0..numberEntries -1) {

            val historyIndex = start + localIndex

            val derivative = gradient[localIndex]

            val newGradientAccumulation = this.decay * this.gradientAccumulation[historyIndex] + this.oneMinusDecay * (derivative * derivative)
            this.gradientAccumulation[historyIndex] = newGradientAccumulation
            val rootMeanSquaredOfDerivatives = FloatMath.sqrt(newGradientAccumulation + this.epsilon)

            val pastUpdateAccumulation = this.updateAccumulation[historyIndex]
            val rootMeanSquaredOfPastUpdates = FloatMath.sqrt(pastUpdateAccumulation + this.epsilon)

            val learningRate = rootMeanSquaredOfPastUpdates / rootMeanSquaredOfDerivatives

            val update = -learningRate * derivative

            this.updateAccumulation[historyIndex] = this.decay * pastUpdateAccumulation + this.oneMinusDecay * (update * update)

            parameters[localIndex] += update

        }

    }

}