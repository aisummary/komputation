package com.komputation.cpu.functions.activation

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test

class ReluTest {

    @Test
    fun test() {

        val result = FloatArray(3)

        relu(floatArrayOf(-1.0f, 0.0f, 1.0f), result, 3)

        val expected = floatArrayOf(0.0f, 0.0f, 1.0f)

        assertArrayEquals(
            result,
            expected)

    }

}