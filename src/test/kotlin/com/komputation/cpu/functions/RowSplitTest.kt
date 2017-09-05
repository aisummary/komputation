package com.komputation.cpu.functions

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class RowSplitTest {

    @Test
    fun test1() {

        val expected = arrayOf(floatArrayOf(1.0f))
        val actual = arrayOf(FloatArray(1))

        splitRows(1, 1, floatArrayOf(1.0f), intArrayOf(1), 1, actual)

        check(actual, expected)

    }

    @Test
    fun test2() {

        val expected = arrayOf(floatArrayOf(1.0f), floatArrayOf(2.0f))
        val actual = arrayOf(FloatArray(1), FloatArray(1))

        splitRows(1, 1, floatArrayOf(1.0f, 2.0f), intArrayOf(1, 1), 2, actual)

        check(actual, expected)

    }

    private fun check(actual: Array<FloatArray>, expected: Array<FloatArray>) {

        assertEquals(actual.size, expected.size)

        expected.zip(actual).forEach { (expectedArray, actualArray) ->

            assertArrayEquals(expectedArray, actualArray, 0.001f)

        }

    }


}