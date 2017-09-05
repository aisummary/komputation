package com.komputation.cpu.optimization

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test

class StochasticGradientDescentTest {

    @Test
    fun testOneDimension() {

        val sgd = CpuStochasticGradientDescent(0.1f)

        val parameter = floatArrayOf(1.0f)

        sgd.updateDensely(parameter, floatArrayOf(0.2f), 1)

        assertArrayEquals(floatArrayOf(0.98f), parameter)

    }

    @Test
    fun testTwoDimensions() {

        val sgd = CpuStochasticGradientDescent(0.1f)

        val parameter = floatArrayOf(1.0f, 2.0f)

        sgd.updateDensely(parameter, floatArrayOf(0.1f, 0.2f), 2)

        assertArrayEquals(floatArrayOf(0.99f, 1.98f), parameter)

    }

}