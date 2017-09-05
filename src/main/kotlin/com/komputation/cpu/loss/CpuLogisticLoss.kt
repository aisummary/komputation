package com.komputation.cpu.loss

import com.komputation.matrix.FloatMath

class CpuLogisticLoss(
    override val numberInputRows: Int,
    override val numberInputColumns: Int) : CpuLossFunction {

    private val numberInputEntries = this.numberInputRows * this.numberInputColumns

    override val backwardResult = FloatArray(this.numberInputEntries)

    // -log(probability of the correct target)
    override fun forward(predictions: FloatArray, targets : FloatArray): Float {

        var loss = 0.0f

        for (index in 0..this.numberInputEntries-1) {

            val target = targets[index]

            if (target == 1.0f) {

                loss += -FloatMath.log(predictions[index])

            }

        }

        return loss

    }

    // -1/target probability if target = 1.0, 0.0 otherwise
    override fun backward(predictions: FloatArray, targets : FloatArray): FloatArray {

        for(indexEntry in 0..this.numberInputEntries - 1) {

            if (targets[indexEntry] == 1.0f) {

                this.backwardResult[indexEntry] = -1.0f.div(predictions[indexEntry])

            }
            else {

                this.backwardResult[indexEntry] = 0f

            }

        }

        return this.backwardResult

    }

}