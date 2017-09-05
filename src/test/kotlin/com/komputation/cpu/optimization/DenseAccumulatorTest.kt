package com.komputation.cpu.optimization

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test

class DenseAccumulatorTest {

    @Test
    fun test() {

        val accumulator = DenseAccumulator(1)

        accumulator.accumulate(floatArrayOf(1.0f))

        val firstExpected = floatArrayOf(1.0f)
        val firstActual = accumulator.getAccumulation()

        assertArrayEquals(firstExpected, firstActual)

        accumulator.reset()

        accumulator.accumulate(floatArrayOf(1.0f))
        accumulator.accumulate(floatArrayOf(2.0f))

        val secondExpected = floatArrayOf(3.0f)
        val secondActual = accumulator.getAccumulation()

        assertArrayEquals(secondExpected, secondActual)

    }

}

