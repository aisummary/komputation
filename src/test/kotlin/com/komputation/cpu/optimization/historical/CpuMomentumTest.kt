package com.komputation.cpu.optimization.historical

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test

class CpuMomentumTest {

    @Test
    fun name() {

        val momentum = CpuMomentum(0.1f, 0.9f, 1)

        val parameter = floatArrayOf(1.0f)
        val gradientSize = 1

        // history * decay - learning rate * gradient = learning rate * gradient = - 0.1 * 0.1 = -0.01
        momentum.updateDensely(parameter, floatArrayOf(0.1f), gradientSize)

        // history * decay + learning rate * gradient = 0.9 * (-0.01) - 0.1 * 0.1 = -0.009 - 0.01 = -0.019
        momentum.updateDensely(parameter, floatArrayOf(0.1f), gradientSize)

        assertArrayEquals(floatArrayOf(1.0f - 0.01f - 0.019f), parameter)

    }

}