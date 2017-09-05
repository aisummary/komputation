package com.komputation.cpu.optimization

import java.util.*

class DenseAccumulator(private val size : Int) {

    private val accumulation = FloatArray(this.size)

    fun accumulate(gradient: FloatArray) {

        for (index in 0..this.size - 1) {

            this.accumulation[index] += gradient[index]

        }

    }

    fun getAccumulation() = this.accumulation

    fun reset() {

        Arrays.fill(this.accumulation, 0.0f)

    }

}


