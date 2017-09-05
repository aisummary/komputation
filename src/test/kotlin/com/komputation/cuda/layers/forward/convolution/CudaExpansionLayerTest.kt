package com.komputation.cuda.layers.forward.convolution

import jcuda.Pointer
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import com.komputation.cuda.getFloatArray
import com.komputation.cuda.kernels.ForwardKernels
import com.komputation.cuda.setFloatArray
import com.komputation.cuda.setIntArray
import com.komputation.cuda.setUpCudaContext

class CudaExpansionLayerTest {

    @Test
    fun testOneByOneConvolutionOneByOneInput() {

        val filterHeight = 1
        val filterWidth = 1
        val input = floatArrayOf(1f)
        val expected = floatArrayOf(1f)

        testFixedLengthForwardPropagation(filterHeight, filterWidth, 1, 1, 1, input, expected)

    }

    @Test
    fun testIncompleteBatch() {

        val filterHeight = 1
        val filterWidth = 1
        val input = floatArrayOf(1f, Float.NaN)
        val expected = floatArrayOf(1f, Float.NaN)

        testFixedLengthForwardPropagation(filterHeight, filterWidth, 2, 1, 1, input, expected)

    }

    @Test
    fun testOneByOneConvolutionTwoByOneInput() {

        val filterHeight = 1
        val filterWidth = 1
        val input = floatArrayOf(1f, 2f)
        val expected = floatArrayOf(1f, 2f)

        testFixedLengthForwardPropagation(filterHeight, filterWidth, 1, 2, 1, input, expected)

    }

    @Test
    fun testOneByOneConvolutionTwoByTwoInput() {

        val filterHeight = 1
        val filterWidth = 1
        val input = floatArrayOf(1f, 2f, 3f, 4f)
        val expected = floatArrayOf(1f, 2f, 3f, 4f)

        testFixedLengthForwardPropagation(filterHeight, filterWidth, 1, 2, 2, input, expected)

    }

    @Test
    fun testTwoByTwoConvolutionTwoByTwoInput() {

        val filterHeight = 2
        val filterWidth = 2
        val input = floatArrayOf(1f, 2f, 3f, 4f)
        val expected = floatArrayOf(1f, 2f, 3f, 4f)

        testFixedLengthForwardPropagation(filterHeight, filterWidth, 1, 2, 2, input, expected)

    }

    @Test
    fun testTwoByTwoConvolutionThreeByTwoThree() {

        /*
            1 4 7
            2 5 8
            3 6 9

            1, 2, 4, 5
            2, 5, 3, 6
            4, 5, 6, 7
            5, 7, 6, 9

         */

        val filterHeight = 2
        val filterWidth = 2
        val input = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f)
        val expected = floatArrayOf(
            1f, 2f, 4f, 5f,
            2f, 3f, 5f, 6f,
            4f, 5f, 7f, 8f,
            5f, 6f, 8f, 9f)

        testFixedLengthForwardPropagation(filterHeight, filterWidth, 1, 3, 3, input, expected)

    }

    private fun testFixedLengthForwardPropagation(filterHeight: Int, filterWidth: Int, maximumBatchSize : Int, numberInputRows: Int, maximumInputColumns : Int, input: FloatArray, expected: FloatArray) {

        val context = setUpCudaContext()

        val expansionLayer = CudaExpansionLayer(
            null,
            numberInputRows,
            maximumInputColumns,
            filterHeight,
            filterWidth,
            { context.createKernel(ForwardKernels.expansion()) },
            { context.createKernel(ForwardKernels.backwardExpansion()) },
            context.warpSize,
            context.maximumNumberOfThreadsPerBlock)

        val deviceInput = Pointer()
        setFloatArray(input, input.size, deviceInput)

        expansionLayer.acquire(maximumBatchSize)
        val deviceResult = expansionLayer.forward(1, deviceInput, false)

        val actual = getFloatArray(deviceResult, maximumBatchSize * expansionLayer.maximumOutputColumns * expansionLayer.numberOutputRows)

        context.destroy()

        assertArrayEquals(expected, actual, 0.01f)

    }

    @Test
    fun testOneOutOfTwoColumnsInstance() {

        val filterHeight = 1
        val filterWidth = 1
        val input = floatArrayOf(1f, Float.NaN)
        val expected = floatArrayOf(1f, 0f)

        testVariableLengthForwardPropagation(filterHeight, filterWidth, 1, 1, 2, intArrayOf(1), input, expected)

    }

    @Test
    fun testOneOutOfThreeColumnsInstance() {

        val filterHeight = 1
        val filterWidth = 1
        val input = floatArrayOf(1f, Float.NaN, Float.NaN)
        val expected = floatArrayOf(1f, 0f, 0f)

        testVariableLengthForwardPropagation(filterHeight, filterWidth, 1, 1, 3, intArrayOf(1), input, expected)

    }

    private fun testVariableLengthForwardPropagation(filterHeight: Int, filterWidth: Int, maximumBatchSize : Int, numberInputRows: Int, maximumInputColumns : Int, lengths : IntArray, input: FloatArray, expected: FloatArray) {

        val context = setUpCudaContext()

        val expansionLayer = CudaExpansionLayer(
            null,
            numberInputRows,
            maximumInputColumns,
            filterHeight,
            filterWidth,
            { context.createKernel(ForwardKernels.expansion()) },
            { context.createKernel(ForwardKernels.backwardExpansion()) },
            context.warpSize,
            context.maximumNumberOfThreadsPerBlock)

        val deviceLengths = Pointer()
        setIntArray(lengths, lengths.size, deviceLengths)

        val deviceInput = Pointer()
        setFloatArray(input, input.size, deviceInput)

        expansionLayer.acquire(maximumBatchSize)
        val deviceResult = expansionLayer.forward(1, deviceLengths, deviceInput, false)

        val actual = getFloatArray(deviceResult, maximumBatchSize * expansionLayer.maximumOutputColumns * expansionLayer.numberOutputRows)

        expansionLayer.release()

        context.destroy()

        assertArrayEquals(expected, actual, 0.01f)
    }

    @Test
    fun testBackwardOneByOneInputOneByOneFilter() {

        val input = floatArrayOf(1f)
        val chain = floatArrayOf(1f)
        val expected = floatArrayOf(1f)

        testBackward(1, 1, input, 1, 1, chain, expected)

    }

    @Test
    fun testBackwardTwoByTwoInputTwoByTwoFilter() {

        val input = floatArrayOf(1f, 2f, 3f, 4f)
        val chain = floatArrayOf(5f, 6f, 7f, 8f)
        val expected = floatArrayOf(5f, 6f, 7f, 8f)

        testBackward(2, 2, input, 2, 2, chain, expected)

    }

    @Test
    fun testBackwardTwoByTwoInputOneByOneFilter() {

        val input = floatArrayOf(1f, 2f, 3f, 4f)
        val chain = floatArrayOf(5f, 6f, 7f, 8f)
        val expected = floatArrayOf(5f, 6f, 7f, 8f)

        testBackward(2, 2, input, 1, 1, chain, expected)

    }

    @Test
    fun testBackwardThreeByThreeInputTwoByTwoFilter() {

        /*

            1 4 7     1 4   2 5   4 7   5 8
            2 5 8     2 5   3 6   5 8   6 9
            3 6 9

             1  4     2  5     4  7     5  8
             2  5     3  6     5  8     6  9
            10 12    14 16    18 20    22 24
            11 13    15 17    19 21    23 25

        */

        val input = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f)
        val chain = floatArrayOf(
            10f, 11f, 12f, 13f,
            14f, 15f, 16f, 17f,
            18f, 19f, 20f, 21f,
            22f, 23f, 24f, 25f)
        val expected = floatArrayOf(
            10f, // 1
            11f + 14f, // 2
            15f, // 3
            12f + 18f, // 4
            13f + 16f + 19f + 22f, // 5
            17f + 23f, // 6
            20f, // 7
            21f + 24f, // 8
            25f) // 9

        testBackward(3, 3, input, 2, 2, chain, expected)

    }

    @Test
    fun testBackwardWithFilterSizeGreaterThanWarpSize() {

        val context = setUpCudaContext()
        val warpSize = context.warpSize
        context.destroy()

        val filterWidth = 1
        val filterHeight = warpSize + 1

        val filterSize = filterHeight

        val input = FloatArray(filterSize) { 1f }
        val chain = FloatArray(filterSize) { 2f }
        val expected = FloatArray(filterSize) { 2f }

        val numberInputRows = filterHeight
        testBackward(numberInputRows, 1, input, filterHeight, filterWidth, chain, expected)

    }

    private fun testBackward(numberInputRows: Int, maximumInputColumns: Int, input : FloatArray, filterHeight: Int, filterWidth: Int, chain : FloatArray, expected: FloatArray) {

        val context = setUpCudaContext()

        val expansionLayer = CudaExpansionLayer(
            null,
            numberInputRows,
            maximumInputColumns,
            filterHeight,
            filterWidth,
            { context.createKernel(ForwardKernels.expansion()) },
            { context.createKernel(ForwardKernels.backwardExpansion()) },
            context.warpSize,
            context.maximumNumberOfThreadsPerBlock)

        expansionLayer.acquire(1)

        val deviceInput = Pointer()
        setFloatArray(input, input.size, deviceInput)

        val deviceChain = Pointer()
        setFloatArray(chain, chain.size, deviceChain)

        expansionLayer.forward(1, deviceInput, false)
        val deviceBackwardResult = expansionLayer.backward(1, deviceChain)

        val actual = getFloatArray(deviceBackwardResult, numberInputRows * maximumInputColumns)

        expansionLayer.release()

        context.destroy()

        assertArrayEquals(expected, actual)


    }


}