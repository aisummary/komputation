package com.komputation.cuda.layers.forward.normalization

import jcuda.Pointer
import jcuda.jcublas.cublasHandle
import jcuda.runtime.JCuda.cudaFree
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import com.komputation.cuda.getFloatArray
import com.komputation.cuda.setFloatArray
import com.komputation.cuda.setUpCudaContext
import com.komputation.layers.forward.normalization.normalizationLayer
import com.komputation.matrix.FloatMath

class CudaNormalizationLayerTest {

    @Test
    fun testForwardOneRowOneColumn() {

        val input = floatArrayOf(1.0f)
        val expected = floatArrayOf(1.0f)

        testForward(input, 1, 1, 1, 1, expected)

    }

    @Test
    fun testForwardTwoRowsOneColumn() {

        val input = floatArrayOf(1.0f, 1.0f)
        val expected = floatArrayOf(0.5f, 0.5f)

        testForward(input, 1, 1, 2, 1, expected)

    }

    @Test
    fun testForwardThreeRowsOneColumn() {

        val input = floatArrayOf(1.0f, 2.0f, 5.0f)
        val expected = floatArrayOf(0.125f, 0.25f, 0.625f)

        testForward(input, 1, 1, 3, 1, expected)

    }


    @Test
    fun testForwardTwoRowsTwoColumns() {

        val input = floatArrayOf(1.0f, 1.0f, 1.0f, 3.0f)
        val expected = floatArrayOf(0.5f, 0.5f, 0.25f, 0.75f)

        testForward(input, 1, 1, 2, 2, expected)

    }

    @Test
    fun testForwardTwoRowsTwoColumnsIncompleteBatch() {

        val input = floatArrayOf(1.0f, 1.0f, 1.0f, 3.0f, Float.NaN, Float.NaN, Float.NaN, Float.NaN)
        val expected = floatArrayOf(0.5f, 0.5f, 0.25f, 0.75f, Float.NaN, Float.NaN, Float.NaN, Float.NaN)

        testForward(input, 1, 2, 2, 2, expected)

    }

    @Test
    fun testBackwardOneRowOneColumn() {

        val input = floatArrayOf(1.0f)
        val chain = floatArrayOf(1.0f)

        this.backward(input, 1, 1, 1, 1, chain)

    }

    @Test
    fun testBackwardTwoRowsOneColumn1() {

        val input = floatArrayOf(1.0f, 1.0f)
        val chain = floatArrayOf(1.0f, 1.0f)
        val expected = floatArrayOf(0.0f, 0.0f)

        testBackward(input, 1, 1, 2, 1, chain, expected)

    }

    @Test
    fun testBackwardTwoRowsOneColumn2() {

        val input = floatArrayOf(1.0f, 2.0f)
        val chain = floatArrayOf(1.0f, 1.0f)
        val expected = floatArrayOf(0.0f, 0.0f)

        testBackward(input, 1, 1,2, 1, chain, expected)

    }

    @Test
    fun testBackwardTwoRowsOneColumn3() {

        val input = floatArrayOf(1.0f, 1.0f)
        val chain = floatArrayOf(2.0f, 1.0f)
        val sum = 2.0f
        val expectedFirst = (2.0f * 1.0f - 1.0f * 1.0f) / FloatMath.pow(sum, 2.0f)
        val expectedSecond = (-2.0f * 1.0f + 1.0f * 1.0f) / FloatMath.pow(sum, 2.0f)
        val expected = floatArrayOf(expectedFirst, expectedSecond)

        testBackward(input, 1, 1, 2, 1, chain, expected)

    }

    @Test
    fun testBackwardOneRowTwoColumns() {

        val input = floatArrayOf(1.0f, 2.0f)
        val chain = floatArrayOf(1.0f, 2.0f)
        val numberRows = 1
        val numberColumns = 2

        val expected = floatArrayOf(0.0f, 0.0f)

        testBackward(input, 1, 1, numberRows, numberColumns, chain, expected)

    }

    @Test
    fun testBackwardTwoRowsTwoColumns() {

        val input = floatArrayOf(0.0f, 1.0f, 2.0f, 3.0f)
        val chain = floatArrayOf(4.0f, 5.0f, 6.0f, 7.0f)
        val numberRows = 2
        val numberColumns = 2

        val firstSum = 0.0f+1.0f
        val firstSquaredSum = firstSum * firstSum
        val secondSum = 2.0f+3.0f
        val secondSquaredSum = secondSum * secondSum

        val expected = floatArrayOf(
            (4.0f * 1.0f - 5.0f * 1.0f) / firstSquaredSum,
            (-4.0f * 0.0f + 5.0f * 0.0f) / firstSquaredSum,
            (6.0f * 3.0f - 7.0f * 3.0f) / secondSquaredSum,
            (-6.0f * 2.0f + 7.0f * 2.0f) / secondSquaredSum)

        testBackward(input, 1, 1, numberRows, numberColumns, chain, expected)

    }

    private fun testForward(input: FloatArray, batchSize : Int, maximumBatchSize : Int, numberRows : Int, numberColumns : Int, expected: FloatArray) {

        val actual = forward(input, batchSize, maximumBatchSize, numberRows, numberColumns)

        assertArrayEquals(expected, actual, 0.001f)

    }

    private fun forward(input: FloatArray, batchSize : Int, maximumBatchSize : Int, numberRows: Int, numberColumns: Int): FloatArray {

        val context = setUpCudaContext()

        val layer = normalizationLayer(numberRows, numberColumns).buildForCuda(context, cublasHandle())
        layer.acquire(maximumBatchSize)

        val deviceInput = Pointer()
        val numberInstanceEntries = numberRows * numberColumns
        val numberBatchEntries = maximumBatchSize * numberInstanceEntries

        setFloatArray(input, numberBatchEntries, deviceInput)

        val deviceResult = layer.forward(batchSize, deviceInput, false)
        val actual = getFloatArray(deviceResult, numberBatchEntries)

        cudaFree(deviceInput)

        layer.release()

        context.destroy()

        return actual

    }

    private fun testBackward(input: FloatArray, batchSize : Int, maximumBatchSize : Int, numberRows: Int, numberColumns: Int, chain: FloatArray, expected: FloatArray) {

        val actual = backward(input, batchSize, maximumBatchSize, numberRows, numberColumns, chain)

        assertArrayEquals(expected, actual, 0.001f)

    }

    private fun backward(input: FloatArray, batchSize : Int, maximumBatchSize : Int, numberRows: Int, numberColumns: Int, chain: FloatArray): FloatArray {

        val context = setUpCudaContext()

        val numberEntries = numberRows * numberColumns
        val numberBatchEntries = maximumBatchSize * numberEntries

        val deviceInput = Pointer()
        setFloatArray(input, numberBatchEntries, deviceInput)

        val deviceChain = Pointer()
        setFloatArray(chain, numberBatchEntries, deviceChain)

        val layer = normalizationLayer(numberRows, numberColumns).buildForCuda(context, cublasHandle())
        layer.acquire(maximumBatchSize)

        layer.forward(batchSize, deviceInput,true)
        val deviceBackwardResult = layer.backward(batchSize, deviceChain)
        val actual = getFloatArray(deviceBackwardResult, numberBatchEntries)

        layer.release()

        cudaFree(deviceInput)
        cudaFree(deviceChain)

        context.destroy()

        return actual

    }

}