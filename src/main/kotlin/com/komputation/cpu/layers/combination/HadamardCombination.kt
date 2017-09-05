package com.komputation.cpu.layers.combination

import com.komputation.cpu.functions.hadamard
import com.komputation.cpu.layers.CombinationLayer
import com.komputation.layers.Resourceful

class HadamardCombination internal constructor(
    name: String?,
    private val numberRows : Int,
    private val numberColumns : Int) : CombinationLayer(name), Resourceful {

    private val numberEntries = this.numberRows * this.numberColumns

    private var firstInput = FloatArray(0)
    private var secondInput = FloatArray(0)

    private var forwardResult = FloatArray(0)
    private var firstBackwardResult = FloatArray(0)
    private var secondBackwardResult = FloatArray(0)

    override fun acquire(maximumBatchSize: Int) {

        this.forwardResult = FloatArray(this.numberEntries)
        this.firstBackwardResult = FloatArray(this.numberEntries)
        this.secondBackwardResult = FloatArray(this.numberEntries)

    }

    override fun release() {

    }

    override fun forward(first: FloatArray, second: FloatArray): FloatArray {

        this.firstInput = first
        this.secondInput = second

        hadamard(this.firstInput, this.secondInput, this.forwardResult, this.numberEntries)

        return this.forwardResult

    }

    // d f(x) * g(x) / d f(x) = g(x)
    override fun backwardFirst(chain: FloatArray): FloatArray {

        hadamard(chain, this.secondInput, this.firstBackwardResult, this.numberEntries)

        return this.firstBackwardResult

    }

    // d f(x) * g(x) / d g(x) = f(x)
    override fun backwardSecond(chain: FloatArray): FloatArray {

        hadamard(chain, this.firstInput, this.secondBackwardResult, this.numberEntries)

        return this.secondBackwardResult

    }

}

fun hadamardCombination(numberRows: Int, numberColumns: Int) = hadamardCombination(null, numberRows, numberColumns)

fun hadamardCombination(name : String? = null, numberRows: Int, numberColumns: Int) = HadamardCombination(name, numberRows, numberColumns)