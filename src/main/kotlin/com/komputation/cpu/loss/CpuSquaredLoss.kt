package com.komputation.cpu.loss

class CpuSquaredLoss(
    override val numberInputRows: Int,
    override val numberInputColumns: Int) : CpuLossFunction {

    private val numberInputEntries = this.numberInputRows * this.numberInputColumns
    override val backwardResult = FloatArray(this.numberInputEntries)

    override fun forward(predictions: FloatArray, targets : FloatArray): Float {

        var loss = 0.0f

        for (indexRow in 0..this.numberInputEntries-1) {

            val prediction = predictions[indexRow]
            val target = targets[indexRow]

            val difference = prediction - target

            loss += 0.5f * (difference * difference)


        }

        return loss

    }

    // loss = 0.5 (prediction - target)^2 = 0.5 prediction^2 - prediction * target + 0.5 target ^2
    // d loss / d prediction = prediction - target

    override fun backward(predictions: FloatArray, targets : FloatArray): FloatArray {

        for(indexEntry in 0..this.numberInputEntries - 1) {

            this.backwardResult[indexEntry] = predictions[indexEntry] - targets[indexEntry]

        }

        return this.backwardResult

    }

}