package com.komputation.cpu.functions

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test

class StepTest {

    @Test
    fun testGetOneDimensionOneStep() {

        val actual = FloatArray(1)
        getStep(floatArrayOf(1.0f), 0, actual, 1)

        val expected = floatArrayOf(1.0f)

        assertArrayEquals(expected, actual, 0.001f)

    }

    @Test
    fun testGetOneDimensionTwoSteps() {

        val firstActual = FloatArray(1)
        val secondActual = FloatArray(1)

        val entries = floatArrayOf(1.0f, 2.0f)

        val stepSize = 1

        getStep(entries, 0, firstActual, stepSize)
        getStep(entries, 1, secondActual, stepSize)

        val firstExpected = floatArrayOf(1.0f)
        val secondExpected = floatArrayOf(2.0f)

        assertArrayEquals(firstExpected, firstActual, 0.001f)
        assertArrayEquals(secondExpected, secondActual, 0.001f)

    }

    @Test
    fun testGetTwoDimensionsTwoSteps() {

        val firstActual = FloatArray(2)
        val secondActual = FloatArray(2)

        val entries = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f)

        val stepSize = 2

        getStep(entries, 0, firstActual, stepSize)
        getStep(entries, 1, secondActual, stepSize)

        val firstExpected = floatArrayOf(1.0f, 2.0f)
        val secondExpected = floatArrayOf(3.0f, 4.0f)

        assertArrayEquals(firstExpected, firstActual, 0.001f)
        assertArrayEquals(secondExpected, secondActual, 0.001f)

    }

    @Test
    fun testSetOneDimensionOneStep() {

        val actual = FloatArray(1)
        setStep(floatArrayOf(1.0f), 0, actual, 1)

        val expected = floatArrayOf(1.0f)

        assertArrayEquals(expected, actual, 0.001f)

    }

    @Test
    fun testOneDimensionTwoSteps() {

        val actual = FloatArray(2)
        setStep(floatArrayOf(1.0f), 0, actual, 1)
        setStep(floatArrayOf(2.0f), 1, actual, 1)

        val expected = floatArrayOf(1.0f, 2.0f)

        assertArrayEquals(expected, actual, 0.001f)

    }

    @Test
    fun testTwoDimensionsTwoSteps() {

        val actual = FloatArray(4)
        setStep(floatArrayOf(1.0f, 2.0f), 0, actual, 2)
        setStep(floatArrayOf(3.0f, 4.0f), 1, actual, 2)

        val expected = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f)

        assertArrayEquals(expected, actual, 0.001f)

    }


}