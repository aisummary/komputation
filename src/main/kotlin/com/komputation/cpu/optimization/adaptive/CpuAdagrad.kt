package com.komputation.cpu.optimization.adaptive

import com.komputation.cpu.optimization.UpdateRule
import com.komputation.matrix.FloatMath

class CpuAdagrad(private val learningRate: Float, private val epsilon : Float, size : Int) : UpdateRule {

    private val sumOfSquaredDerivatives = FloatArray(size)

    override fun updateSparsely(start : Int, parameters: FloatArray, gradient: FloatArray, numberEntries: Int) {

        for(index in 0..numberEntries -1) {

            val historyIndex = start + index
            val derivative = gradient[index]

            this.updateHistory(historyIndex, derivative)

            val adaptiveLearningRate = this.adaptLearningRate(historyIndex)

            val update = -adaptiveLearningRate * derivative

            parameters[index] += update

        }

    }

    private fun updateHistory(historyIndex: Int, derivative: Float) {

        this.sumOfSquaredDerivatives[historyIndex] += derivative * derivative

    }

    private fun adaptLearningRate(historyIndex: Int) =

        this.learningRate / (FloatMath.sqrt(this.sumOfSquaredDerivatives[historyIndex]) + this.epsilon)

}