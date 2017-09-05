package com.komputation.cpu.optimization.historical

import com.komputation.cpu.optimization.UpdateRule

/*
    After n step, the first scaled gradient is decayed by momentum^(n-1), the second scaled gradient is decayed by momentum^(n-2), etc.
    history_0 = 0
    history_1 = momentum * history_0 - learning_rate * gradient_1
              = - learning_rate * gradient_1
    history_2 = momentum * history_1 - learning_rate * gradient_2
              = momentum * (- learning_rate * gradient_1) - learning_rate * gradient_2
              = - momentum * learning_rate * gradient_1 - learning_rate * gradient_2
    history_3 = momentum * history_2 - learning_rate * gradient_3
              = momentum * (- momentum * learning_rate * gradient_1 - learning_rate * gradient_2) - learning_rate * gradient_3
              = - momentum^2 * learning_rate * gradient_1 - momentum * learning_rate * gradient 2 - learning_rate * gradient_3
 */
class CpuMomentum(private val learningRate: Float, private val momentum: Float, historySize: Int) : UpdateRule {

    private val history = FloatArray(historySize)

    override fun updateSparsely(start : Int, parameters: FloatArray, gradient: FloatArray, numberEntries: Int) {

        for(localIndex in 0..numberEntries - 1) {

            val historyIndex = start + localIndex

            val derivative = gradient[localIndex]

            parameters[localIndex] += this.updateHistory(derivative, historyIndex)

        }

    }

    private fun updateHistory(derivative: Float, historyIndex : Int): Float {

        val newStateEntry = this.momentum * this.history[historyIndex] - this.learningRate * derivative

        this.history[historyIndex] = newStateEntry

        return newStateEntry

    }

}