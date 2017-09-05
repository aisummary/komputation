package com.komputation.cpu.optimization

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class SparseAccumulatorTest {

    @Test
    fun testOneDimension() {

        val accumulator = SparseAccumulator(1, 1, 1, 1)

        accumulator.accumulate(intArrayOf(0), 1, floatArrayOf(1.0f))

        check(accumulator, 1, intArrayOf(0), floatArrayOf(1.0f), arrayOf(floatArrayOf(1.0f)))

        accumulator.accumulate(intArrayOf(0), 1, floatArrayOf(2.0f))

        check(accumulator, 1, intArrayOf(0), floatArrayOf(2.0f), arrayOf(floatArrayOf(3.0f)))

    }

    @Test
    fun testTwoDimensions() {

        val accumulator = SparseAccumulator(1, 1, 1, 2)

        accumulator.accumulate(intArrayOf(0), 1, floatArrayOf(1.1f, 1.2f))

        check(accumulator, 1, intArrayOf(0), floatArrayOf(1.0f), arrayOf(floatArrayOf(1.1f, 1.2f)))

        accumulator.accumulate(intArrayOf(0), 1, floatArrayOf(1.0f, 2.0f))

        check(accumulator, 1, intArrayOf(0), floatArrayOf(2.0f), arrayOf(floatArrayOf(2.1f, 3.2f)))

    }

    @Test
    fun testTwoVectorsTwoDimensionsAndBatchesOfTwo() {

        val accumulator = SparseAccumulator(2, 2, 1, 2)

        accumulator.accumulate(intArrayOf(0), 1, floatArrayOf(1.1f, 1.2f))
        accumulator.accumulate(intArrayOf(1), 1, floatArrayOf(2.1f, 2.2f))

        check(accumulator, 2, intArrayOf(0, 1), floatArrayOf(1.0f, 1.0f), arrayOf(floatArrayOf(1.1f, 1.2f), floatArrayOf(2.1f, 2.2f)))

    }

    @Test
    fun testTwoVectorsTwoDimensionsAndLengthsOfTwo() {

        val accumulator = SparseAccumulator(2, 1, 2, 2)

        accumulator.accumulate(intArrayOf(0, 1), 2, floatArrayOf(1.1f, 1.2f, 2.1f, 2.2f))

        check(accumulator, 2, intArrayOf(0, 1), floatArrayOf(1.0f, 1.0f), arrayOf(floatArrayOf(1.1f, 1.2f), floatArrayOf(2.1f, 2.2f)))

    }

    @Test
    fun testTwoVectorsTwoDimensionsBatchesOfTwoAndLengthsOfTwo() {

        val accumulator = SparseAccumulator(2, 1, 2, 2)

        accumulator.accumulate(intArrayOf(0, 1), 2, floatArrayOf(1.1f, 1.2f, 2.1f, 2.2f))
        accumulator.accumulate(intArrayOf(0, 1), 2, floatArrayOf(1.3f, 1.4f, 2.3f, 2.4f))

        check(accumulator, 2, intArrayOf(0, 1), floatArrayOf(2.0f, 2.0f), arrayOf(floatArrayOf(2.4f, 2.6f), floatArrayOf(4.4f, 4.6f)))

    }

    @Test
    fun testTwoVectorsTwoDimensionsBatchesOfTwoAndLengthsOfTwoReversed() {

        val accumulator = SparseAccumulator(2, 1, 2, 2)

        accumulator.accumulate(intArrayOf(0, 1), 2, floatArrayOf(1.1f, 1.2f, 2.1f, 2.2f))
        accumulator.accumulate(intArrayOf(1, 0), 2, floatArrayOf(2.3f, 2.4f, 1.3f, 1.4f))

        check(accumulator, 2, intArrayOf(0, 1), floatArrayOf(2.0f, 2.0f), arrayOf(floatArrayOf(2.4f, 2.6f), floatArrayOf(4.4f, 4.6f)))

    }

    @Test
    fun testRepeatOnce() {

        val accumulator = SparseAccumulator(1, 1, 2, 1)

        accumulator.accumulate(intArrayOf(0, 0), 2, floatArrayOf(1.0f, 2.0f))

        check(accumulator, 1, intArrayOf(0), floatArrayOf(1.0f), arrayOf(floatArrayOf(3.0f)))

    }

    @Test
    fun testRepeatTwice() {

        val accumulator = SparseAccumulator(1, 2, 2, 1)

        accumulator.accumulate(intArrayOf(0, 0), 2, floatArrayOf(1.0f, 2.0f))
        accumulator.accumulate(intArrayOf(0, 0), 2, floatArrayOf(3.0f, 4.0f))

        check(accumulator, 1, intArrayOf(0), floatArrayOf(2.0f), arrayOf(floatArrayOf(10.0f)))

    }

    private fun check(
        accumulator: SparseAccumulator,
        expectedSize: Int,
        expectedIds: IntArray,
        expectedCounts: FloatArray,
        expectedSums: Array<FloatArray>) {

        val actualSize = accumulator.getSize()
        val actualIds = accumulator.getIds()
        val actualCounts = accumulator.getCounts()
        val actualSums = accumulator.getSums()

        assertEquals(expectedSize, actualSize)

        for (index in 0..actualSize - 1) {

            assertEquals(expectedIds[index], actualIds[index])
            assertEquals(expectedCounts[index], actualCounts[index])
            assertArrayEquals(actualSums[index], expectedSums[index], 0.001f)

        }



    }

}