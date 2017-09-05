package com.komputation.cpu.optimization.adaptive

import com.komputation.cpu.optimization.UpdateRule
import com.komputation.matrix.FloatMath

class CpuRMSProp(private val learningRate : Float, private val decay : Float, private val epsilon : Float, size : Int) : UpdateRule {

    private val oneMinusDecay = 1.0f - this.decay

    private val accumulation = FloatArray(size)

    override fun updateSparsely(start : Int, parameters: FloatArray, gradient: FloatArray, numberEntries: Int) {

        for(index in 0..numberEntries -1) {

            val derivative = gradient[index]

            val historyIndex = start + index

            val newAccumulation = this.decay * this.accumulation[historyIndex] + this.oneMinusDecay * (derivative * derivative)
            this.accumulation[historyIndex] = newAccumulation

            val adaptiveLearningRate = this.learningRate / FloatMath.sqrt(newAccumulation + this.epsilon)

            val update = -adaptiveLearningRate * derivative

            parameters[index] += update

        }

    }

}