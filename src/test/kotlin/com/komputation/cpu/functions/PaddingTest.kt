package com.komputation.cpu.functions

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test

class PaddingTest {

    @Test
    fun testNoDimensionToBePadded() {

        val input = intArrayOf(1)
        val maximumLength = 1
        val expected = intArrayOf(1)

        test(input, maximumLength, expected)

    }

    @Test
    fun testOneDimensionToBePadded() {

        val input = intArrayOf(1)
        val maximumLength = 2
        val expected = intArrayOf(1, -1)

        test(input, maximumLength, expected)

    }

    @Test
    fun testTwoDimensionsToBePadded() {

        val input = intArrayOf(1)
        val maximumLength = 3
        val expected = intArrayOf(1, -1, -1)

        test(input, maximumLength, expected)

    }

    private fun test(input : IntArray, maximumLength : Int, expected : IntArray, symbol : Int = -1) {

        val actual = IntArray(maximumLength)

        pad(input, input.size, maximumLength, symbol, actual)

        assertArrayEquals(expected, actual)

    }

}