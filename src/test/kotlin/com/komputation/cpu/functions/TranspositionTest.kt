package com.komputation.cpu.functions

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test

class TranspositionTest {

    @Test
    fun test1() {

        val actual = FloatArray(1)
        transpose(1, 1, floatArrayOf(1.0f), actual)
        val expected = floatArrayOf(1.0f)

        assertArrayEquals(expected, actual)

    }

    @Test
    fun test2() {

        val actual = FloatArray(2)
        transpose(2, 1, floatArrayOf(1.0f, 2.0f), actual)
        val expected = floatArrayOf(1.0f, 2.0f)

        assertArrayEquals(expected, actual)

    }

    @Test
    fun test3() {

        val actual = FloatArray(2)
        transpose(1, 2, floatArrayOf(1.0f, 2.0f), actual)
        val expected = floatArrayOf(1.0f, 2.0f)

        assertArrayEquals(expected, actual)

    }

    @Test
    fun test4() {

        val actual = FloatArray(4)
        transpose(2, 2, floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f), actual)
        val expected = floatArrayOf(1.0f, 3.0f, 2.0f, 4.0f)

        assertArrayEquals(expected, actual)

    }

    @Test
    fun test5() {

        val actual = FloatArray(6)
        transpose(3, 2, floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f), actual)
        val expected = floatArrayOf(1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f)

        assertArrayEquals(expected, actual)

    }

    @Test
    fun test6() {

        val actual = FloatArray(6)
        transpose(2, 3, floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f), actual)
        val expected = floatArrayOf(1.0f, 3.0f, 5.0f, 2.0f, 4.0f, 6.0f)

        assertArrayEquals(expected, actual)

    }

}