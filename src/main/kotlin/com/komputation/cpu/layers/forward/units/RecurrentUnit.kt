package com.komputation.cpu.layers.forward.units

abstract class RecurrentUnit (val name : String?) {

    abstract fun forwardStep(withinBatch : Int, indexStep: Int, state: FloatArray, input: FloatArray, isTraining : Boolean) : FloatArray

    abstract fun backwardStep(withinBatch : Int, step : Int, chain : FloatArray): Pair<FloatArray, FloatArray>

    abstract fun backwardSeries()

}