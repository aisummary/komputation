package com.komputation.cuda.layers.forward.maxpooling

import jcuda.Pointer
import jcuda.jcublas.cublasHandle
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import com.komputation.cuda.getFloatArray
import com.komputation.cuda.setFloatArray
import com.komputation.cuda.setIntArray
import com.komputation.cuda.setUpCudaContext
import com.komputation.layers.forward.convolution.MaxPoolingLayer
import com.komputation.layers.forward.convolution.maxPoolingLayer
import java.util.*

class CudaMaxPoolingLayerTest {

    @Test
    fun testForwardOneRowOneColumn() {

        val input = floatArrayOf(1f)
        val expected = floatArrayOf(1f)

        testFixedLengthForward(1, 1, input, expected)

    }

    @Test
    fun testForwardOneRowTwoColumns() {

        val input = floatArrayOf(1f, 2f)
        val expected = floatArrayOf(2f)

        testFixedLengthForward(1, 2, input, expected)

    }

    @Test
    fun testForwardOneRowTwoColumnsReversed() {

        val input = floatArrayOf(2f, 1f)
        val expected = floatArrayOf(2f)

        testFixedLengthForward(1, 2, input, expected)

    }

    @Test
    fun testForwardOneRowThreeColumns() {

        val input = floatArrayOf(1f, 3f, 2f)
        val expected = floatArrayOf(3f)

        testFixedLengthForward(1,3, input, expected)

    }

    @Test
    fun testForwardOneRowThirtyThree() {

        val input = FloatArray(33) { index -> (index+1).toFloat() }
        val expected = floatArrayOf(33f)

        testFixedLengthForward(1,33, input, expected)

    }

    @Test
    fun testForwardTwoRowsOneColumn() {

        val input = floatArrayOf(1f, 2f)
        val expected = floatArrayOf(1f, 2f)

        testFixedLengthForward(2,1, input, expected)

    }

    @Test
    fun testForwardTwoRowsTwoColumns() {

        /*
            1.0 3.0
            2.0 -4.0
        */
        val input = floatArrayOf(1f, 2f, 3f, -4f)
        val expected = floatArrayOf(3f, 2f)

        testFixedLengthForward(2,2, input, expected)

    }

    @Test
    fun testForwardOneRowOneOutOfTwoColumns() {

        val input = floatArrayOf(1f, Float.NaN)
        val expected = floatArrayOf(1f)

        testVariableLengthForward(1, 2, intArrayOf(1), input, expected)

    }

    @Test
    fun testBackwardOneRowOneColumn() {

        testBackward(1, 1, 1, floatArrayOf(2f), floatArrayOf(3f), floatArrayOf(3f))

    }

    @Test
    fun testBackwardOneRowTwoColumns() {

        testBackward(1, 2, 2, floatArrayOf(1f, 2f), floatArrayOf(3f), floatArrayOf(0f, 3f))

    }


    @Test
    fun testBackwardOneRowTwoColumnsReversedInput() {

        testBackward(1, 2, 2, floatArrayOf(2f, 1f), floatArrayOf(3f), floatArrayOf(3f, 0f))

    }

    @Test
    fun testBackwardTwoRowsOneColumn() {

        testBackward(2, 1, 1, floatArrayOf(1f, 2f), floatArrayOf(3f, 4f), floatArrayOf(3f, 4f))

    }

    @Test
    fun testBackwardTwoRowsTwoColumns() {

        /*
            1 4 => 0 5
            2 3    0 6
         */

        testBackward(2, 2, 2, floatArrayOf(1f, 2f, 4f, 3f), floatArrayOf(5f, 6f), floatArrayOf(0f, 0f, 5f, 6f))

    }

    @Test
    fun testBackwardTwoRowsTwoColumnsTransposedInput() {

        /*
            1 2 => 0 5
            4 3    6 0
         */

        testBackward(2, 2, 2, floatArrayOf(1f, 4f, 2f, 3f), floatArrayOf(5f, 6f), floatArrayOf(0f, 6f, 5f, 0f))

    }

    @Test
    fun testBackwardTwoRowsThreeColumns() {

        /*
            -1 2 3     0 0 10
            4 -5 -6 => 11 0 0
            -7 8 -9    0 12 0
         */

        testBackward(3, 3, 3, floatArrayOf(-1f, 4f, -7f, 2f, -5f, 8f, 3f, -6f, -9f), floatArrayOf(10f, 11f, 12f), floatArrayOf(0f, 11f, 0f, 0f, 0f, 12f, 10f, 0f, 0f))

    }

    @Test
    fun testBackwardOneRowOneColumnOutOfTwoColumns() {

        testBackward(1, 1, 2, floatArrayOf(2f), floatArrayOf(3f), floatArrayOf(3f))

    }

    private fun testFixedLengthForward(numberRows : Int, maximumNumberColumns : Int, input : FloatArray, expected : FloatArray) {

        val batchSize = 1

        val cudaContext = setUpCudaContext()

        val maxPoolingLayer = maxPoolingLayer(numberRows, maximumNumberColumns).buildForCuda(cudaContext, cublasHandle())

        maxPoolingLayer.acquire(batchSize)

        val deviceInput = Pointer()
        setFloatArray(input, numberRows * maximumNumberColumns, deviceInput)

        val deviceResult = maxPoolingLayer.forward(batchSize, deviceInput, false)

        val actual = getFloatArray(deviceResult, numberRows)

        cudaContext.destroy()

        assertArrayEquals(expected, actual, 0.01f)

    }

    private fun testVariableLengthForward(numberRows : Int, maximumNumberColumns : Int, lengths : IntArray, input : FloatArray, expected : FloatArray) {

        val batchSize = 1

        val cudaContext = setUpCudaContext()

        val maxPoolingLayer = maxPoolingLayer(numberRows, maximumNumberColumns).buildForCuda(cudaContext, cublasHandle())

        maxPoolingLayer.acquire(batchSize)

        val deviceLengths = Pointer()
        setIntArray(lengths, lengths.size, deviceLengths)

        val deviceInput = Pointer()
        setFloatArray(input, numberRows * maximumNumberColumns, deviceInput)

        val deviceResult = maxPoolingLayer.forward(batchSize, deviceLengths, deviceInput, false)

        val actual = getFloatArray(deviceResult, numberRows)

        cudaContext.destroy()

        assertArrayEquals(expected, actual, 0.01f)

    }

    private fun testBackward(numberRows : Int, numberColumns : Int, maximumNumberColumns: Int, input: FloatArray, chain : FloatArray, expected: FloatArray) {

        val batchSize = 1

        val cudaContext = setUpCudaContext()

        val maxPoolingLayer = maxPoolingLayer(numberRows, maximumNumberColumns).buildForCuda(cudaContext, cublasHandle())
        maxPoolingLayer.acquire(batchSize)

        val deviceInput = Pointer()
        setFloatArray(input, numberRows * maximumNumberColumns, deviceInput)

        val deviceChain = Pointer()
        setFloatArray(chain, numberRows, deviceChain)

        maxPoolingLayer.forward(batchSize, deviceInput,false)

        val deviceBackwardResult = maxPoolingLayer.backward(batchSize, deviceChain)

        val actual = getFloatArray(deviceBackwardResult, numberRows * numberColumns)

        cudaContext.destroy()

        assertArrayEquals(expected, actual, 0.01f)

    }

    @Test
    fun testComparison() {

        val random = Random()

        val maximumBatchSize = 1

        repeat(5) {

            val cudaContext = setUpCudaContext()

            val numberRows = random.nextInt(100) + 1
            val maximumColumns = random.nextInt(100) + 1

            val maxPoolingLayer = MaxPoolingLayer(null, numberRows, 1, maximumColumns, Float.NaN)

            val cudaLayer = maxPoolingLayer.buildForCuda(cudaContext, cublasHandle())
            val cpuLayer = maxPoolingLayer.buildForCpu()

            val input = FloatArray(numberRows * maximumColumns) { Float.NaN }

            val numberColumns = random.nextInt(maximumColumns) + 1

            for (indexColumn in 0..numberColumns - 1) {

                val startColumn = indexColumn * numberRows

                for (indexRow in 0..numberRows - 1) {

                    input[startColumn + indexRow] = random.nextFloat()

                }

            }

            cpuLayer.acquire(1)
            cudaLayer.acquire(1)

            val cpuResult = cpuLayer.forward(0, numberColumns, input, false)

            val deviceInput = Pointer()
            setFloatArray(input, input.size, deviceInput)
            val deviceLengths = Pointer()
            setIntArray(IntArray(maximumBatchSize) { numberColumns }, maximumBatchSize, deviceLengths)

            val deviceCudaResult = cudaLayer.forward(1, deviceLengths, deviceInput, false)
            val cudaResult = getFloatArray(deviceCudaResult, numberRows)

            for (indexEntry in 0..numberRows-1) {

                assertEquals(cpuResult[indexEntry], cudaResult[indexEntry], 0.00001f)

            }

            cpuLayer.release()
            cudaLayer.release()

            cudaContext.destroy()

        }

    }

}