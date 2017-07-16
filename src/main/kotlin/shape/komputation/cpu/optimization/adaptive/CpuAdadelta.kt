package shape.komputation.cpu.optimization.adaptive

import shape.komputation.cpu.optimization.UpdateRule


class CpuAdadelta(private val decay : Double, private val epsilon : Double, size : Int) : UpdateRule {

    private val oneMinusDecay = 1.0 - this.decay

    private val gradientAccumulation = DoubleArray(size)
    private val updateAccumulation = DoubleArray(size)

    override fun updateSparsely(start : Int, parameters: DoubleArray, gradient: DoubleArray, gradientSize : Int) {

        for(localIndex in 0..gradientSize-1) {

            val historyIndex = start + localIndex

            val derivative = gradient[localIndex]

            val newGradientAccumulation = this.decay * this.gradientAccumulation[historyIndex] + this.oneMinusDecay * Math.pow(derivative, 2.0)
            this.gradientAccumulation[historyIndex] = newGradientAccumulation
            val rootMeanSquaredOfDerivatives = Math.sqrt(newGradientAccumulation + this.epsilon)

            val pastUpdateAccumulation = this.updateAccumulation[historyIndex]
            val rootMeanSquaredOfPastUpdates = Math.sqrt(pastUpdateAccumulation + this.epsilon)

            val learningRate = rootMeanSquaredOfPastUpdates / rootMeanSquaredOfDerivatives

            val update = -learningRate * derivative

            this.updateAccumulation[historyIndex] = this.decay * pastUpdateAccumulation + this.oneMinusDecay * Math.pow(update, 2.0)

            parameters[localIndex] += update

        }

    }

}