package com.komputation.cpu.layers.combination

import com.komputation.cpu.functions.add
import com.komputation.cpu.layers.CombinationLayer
import com.komputation.layers.Resourceful

/*
   Ex:
   f(x) = (g(x)+h(x))^2 = g(x)^2 + 2*g(x)*h(x) + h(x)^2
   d f(x) / g(x) = 2*g(x) + 2*h(x)
   d f(x) / h(x) = 2*h(x) + 2*g(x)
   chain = d (g(x)+h(x))^2 / d g(x)+h(x) = 2 * (g(x)+h(x)) = 2*g(x) + 2*h(x)
   chain * d g(x)+h(x) / d g(x) = chain
   chain * d g(x)+h(x) / d h(x) = chain
*/
class AdditionCombination internal constructor(
    name : String? = null,
    private val numberRows : Int,
    private val numberColumns : Int) : CombinationLayer(name), Resourceful {

    private val numberEntries = this.numberRows * this.numberColumns
    private var forwardResult = FloatArray(0)

    override fun acquire(maximumBatchSize: Int) {

        this.forwardResult = FloatArray(this.numberEntries)

    }

    override fun release() {

    }

    override fun forward(first: FloatArray, second: FloatArray): FloatArray {

        add(first, second, this.forwardResult, this.numberEntries)

        return this.forwardResult

    }

    // d (x + y) / d x = 1
    override fun backwardFirst(chain: FloatArray) =

        chain

    // d (x + y) / d y = 1
    override fun backwardSecond(chain: FloatArray) =

        chain

}

fun additionCombination(numberRows : Int, numberColumns : Int) =

    AdditionCombination(null, numberRows, numberColumns)

fun additionCombination(name : String? = null, numberRows : Int, numberColumns : Int) =

    AdditionCombination(name, numberRows, numberColumns)