package com.komputation.matrix

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test

class PartitioningTest {

    @Test
    fun test1() {

        val actual = partitionIndices(2, 1)
        val expected = arrayOf(intArrayOf(0), intArrayOf(1))

        assertArrayEquals(expected, actual)

    }


    @Test
    fun test2() {

        val actual = partitionIndices(2, 2)
        val expected = arrayOf(intArrayOf(0, 1))

        assertArrayEquals(expected, actual)

    }

    @Test
    fun test3() {

        val actual = partitionIndices(2, 3)
        val expected = arrayOf(intArrayOf(0, 1))

        assertArrayEquals(expected, actual)

    }

}