package com.komputation.cpu.functions

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class FindMaxIndicesInRowsTest {

    @Test
    fun testOneByOne() {

        val expected = intArrayOf(0)

        val actual = IntArray(1)
        findMaxIndicesInRows(floatArrayOf(1.0f), 1, 1, actual)

        assertArrayEquals(expected, actual)

    }

    @Test
    fun testTwoByOne() {

        val expected = intArrayOf(0, 1)
        val actual = IntArray(2)
        findMaxIndicesInRows(floatArrayOf(1.0f, 1.0f), 2, 1, actual)

        assertArrayEquals(expected, actual)

    }

    @Test
    fun testTwoByTwo() {

        val expected = intArrayOf(2, 3)
        val actual = IntArray(2)
        findMaxIndicesInRows(floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f), 2, 2, actual)

        assertArrayEquals(expected, actual)

    }

    @Test
    fun testTwoByTwoReversed() {

        val expected = intArrayOf(0, 1)
        val actual = IntArray(2)
        findMaxIndicesInRows(floatArrayOf(4.0f, 3.0f, 2.0f, 1.0f), 2, 2, actual)

        assertArrayEquals(expected, actual)

    }

}

class FindMaxIndicesInColumnsTest {

    @Test
    fun testOneDimension() {

        assertEquals(0, findMaxIndex(floatArrayOf(1f), 0, 1))


    }

    @Test
    fun testTwoDimensions() {

        assertEquals(0, findMaxIndex(floatArrayOf(2f, 0f), 0, 2))
        assertEquals(1, findMaxIndex(floatArrayOf(1f, 2f), 0, 2))


    }

    @Test
    fun testOffset() {

        assertEquals(0, findMaxIndex(floatArrayOf(2f, 0f, 1f), 0, 3))
        assertEquals(1, findMaxIndex(floatArrayOf(2f, 0f, 1f), 1, 2))
        assertEquals(0, findMaxIndex(floatArrayOf(2f, 1f, 0f), 0, 2))

    }

    @Test
    fun testLimit() {

        assertEquals(2, findMaxIndex(floatArrayOf(0f, 1f, 2f), 0, 3))
        assertEquals(1, findMaxIndex(floatArrayOf(0f, 1f, 2f), 0, 2))
        assertEquals(0, findMaxIndex(floatArrayOf(0f, 1f, 2f), 0, 1))

    }


}