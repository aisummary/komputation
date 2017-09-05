package com.komputation.cpu.functions

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test

class RowSummationTest {

    @Test
    fun test1() {

        val actual = FloatArray(1)
        sumRows(1, 1, floatArrayOf(1.0f), actual)
        val expected = floatArrayOf(1.0f)

        assertArrayEquals(expected, actual)

    }

    @Test
    fun test2() {

        val actual = FloatArray(2)
        sumRows(2, 1, floatArrayOf(1.0f, 2.0f), actual)
        val expected = floatArrayOf(1.0f, 2.0f)

        assertArrayEquals(expected, actual)

    }

    @Test
    fun test3() {

        val actual = FloatArray(2)
        sumRows(2, 2, floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f), actual)
        val expected = floatArrayOf(4.0f, 6.0f)

        assertArrayEquals(expected, actual)

    }

}