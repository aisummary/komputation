package com.komputation.cpu.layers.entry

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import com.komputation.layers.entry.lookupLayer
import com.komputation.matrix.intMatrix

class CpuLookupLayerTest {

    @Test
    fun testOneDimension() {

        val firstVector = floatArrayOf(1.0f)

        val lookupLayer = lookupLayer(arrayOf(firstVector), 1, false, 1).buildForCpu()

        val expected = floatArrayOf(1.0f)
        lookupLayer.forward(intMatrix(0))
        val actual = lookupLayer.forwardResult

        assertArrayEquals(expected, actual, 0.001f)

    }

    @Test
    fun testTwoDimensions() {

        val lookupLayer = lookupLayer(arrayOf(floatArrayOf(1.0f, 2.0f)), 1, false, 2).buildForCpu()

        val expected = floatArrayOf(1.0f, 2.0f)
        lookupLayer.forward(intMatrix(0))
        val actual = lookupLayer.forwardResult

        assertArrayEquals(expected, actual, 0.001f)

    }

    @Test
    fun testOneOutOfTwoVectors() {

        val lookupLayer = lookupLayer(arrayOf(floatArrayOf(1.0f), floatArrayOf(2.0f)), 1, false, 1).buildForCpu()

        lookupLayer.forward(intMatrix(0))

        assertArrayEquals(
            floatArrayOf(1.0f),
            lookupLayer.forwardResult,
            0.001f)

        lookupLayer.forward(intMatrix(1))

        assertArrayEquals(
            floatArrayOf(2.0f),
            lookupLayer.forwardResult,
            0.001f)

    }

    @Test
    fun testTwoOutOfTwoVectors() {

        val lookupLayer = lookupLayer(arrayOf(floatArrayOf(1.0f), floatArrayOf(2.0f)), 2, false, 1).buildForCpu()

        lookupLayer.forward(intMatrix(0, 1))

        assertArrayEquals(
            floatArrayOf(1.0f, 2.0f),
            lookupLayer.forwardResult,
            0.001f)

    }

    @Test
    fun testTwoOutOfTwoVectorsTwoDimensions() {

        val lookupLayer = lookupLayer(arrayOf(floatArrayOf(1.0f, 2.0f), floatArrayOf(3.0f, 4.0f)), 2, false, 2).buildForCpu()

        lookupLayer.forward(intMatrix(0, 1))

        assertArrayEquals(
            floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f),
            lookupLayer.forwardResult,
            0.001f)

    }

    @Test
    fun testTwoOutOfTwoVectorsTwoDimensionsReversed() {

        val lookupLayer = lookupLayer(arrayOf(floatArrayOf(1.0f, 2.0f), floatArrayOf(3.0f, 4.0f)), 2, false, 2).buildForCpu()

        lookupLayer.forward(intMatrix(1, 0))

        assertArrayEquals(
            floatArrayOf(3.0f, 4.0f, 1.0f, 2.0f),
            lookupLayer.forwardResult,
            0.001f)

    }

}