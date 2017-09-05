package com.komputation.cpu.functions.activation

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test

class NormalizationTest {

    @Test
    fun testOneRowOneColumn () {

        val input = floatArrayOf(1.0f)

        val actual = FloatArray(1)
        normalize(1, 1, input, FloatArray(1), actual)

        val expected = floatArrayOf(1.0f)

        assertArrayEquals(expected, actual, 1.0f)

    }

    @Test
    fun testTwoRowsOneColumn () {

        val input = floatArrayOf(1.0f, 3.0f)

        val actual = FloatArray(2)
        normalize(2, 1, input, FloatArray(1), actual)

        val expected = floatArrayOf(0.25f, 0.75f)

        assertArrayEquals(expected, actual, 1.0f)

    }

    @Test
    fun testTwoRowsTwoColumns () {

        val input = floatArrayOf(1.0f, 3.0f, 2.0f, 8.0f)

        val actual = FloatArray(4)
        normalize(2, 2, input, FloatArray(2), actual)

        val expected = floatArrayOf(0.25f, 0.75f, 0.2f, 0.8f)

        assertArrayEquals(expected, actual, 1.0f)

    }


}