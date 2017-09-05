package com.komputation.cpu.functions

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test

class RowStackingTest {

    @Test
    fun test() {

        val firstRow = floatArrayOf(1.0f, 2.0f)
        val secondRow = floatArrayOf(3.0f, 4.0f)

        val actual = FloatArray(4)
        stackRows(intArrayOf(1, 1), 2, 2, actual, firstRow, secondRow)
        val expected = floatArrayOf(1f, 3f, 2f, 4f)

        assertArrayEquals(expected, actual, 0.01f)

    }

}