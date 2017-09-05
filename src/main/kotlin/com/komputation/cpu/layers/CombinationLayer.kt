package com.komputation.cpu.layers

abstract class CombinationLayer(private val name: String?) {

    abstract fun forward(first: FloatArray, second: FloatArray) : FloatArray

    abstract fun backwardFirst(chain : FloatArray) : FloatArray

    abstract fun backwardSecond(chain : FloatArray) : FloatArray

}