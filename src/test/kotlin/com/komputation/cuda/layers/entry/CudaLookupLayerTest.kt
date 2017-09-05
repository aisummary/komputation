package com.komputation.cuda.layers.entry

import jcuda.Pointer
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import com.komputation.cuda.getFloatArray
import com.komputation.cuda.memory.InputMemory
import com.komputation.cuda.setFloatArray
import com.komputation.cuda.setUpCudaContext
import com.komputation.layers.entry.lookupLayer
import com.komputation.matrix.Matrix
import com.komputation.matrix.intMatrix

class CudaLookupLayerTest {

    @Test
    fun testOneVectorOneDimension() {

        val vectors = arrayOf(floatArrayOf(1f))
        val dimension = 1
        val batch = intArrayOf(0)
        val maximumLength = 1
        val inputs : Array<Matrix> = arrayOf(intMatrix(0))
        val expected = floatArrayOf(1f)

        testForward(vectors, dimension, batch, true, maximumLength, inputs, expected)

    }

    @Test
    fun testOneVectorTwoDimensions() {

        val vectors = arrayOf(floatArrayOf(1f, 2f))
        val dimension = 2
        val batch = intArrayOf(0)
        val maximumLength = 1
        val inputs : Array<Matrix> = arrayOf(intMatrix(0))
        val expected = floatArrayOf(1f, 2f)

        testForward(vectors, dimension, batch, true, maximumLength, inputs, expected)

    }

    @Test
    fun testTwoVectorsOneDimension() {

        val vectors = arrayOf(floatArrayOf(1f), floatArrayOf(2f))
        val dimension = 1
        val batch = intArrayOf(0)
        val maximumLength = 2
        val inputs : Array<Matrix> = arrayOf(intMatrix(0, 1))
        val expected = floatArrayOf(1f, 2f)

        testForward(vectors, dimension, batch, true, maximumLength, inputs, expected)

    }

    @Test
    fun testTwoVectorsOneDimensionReversed() {

        val vectors = arrayOf(floatArrayOf(1f), floatArrayOf(2f))
        val dimension = 1
        val batch = intArrayOf(0)
        val maximumLength = 2
        val inputs : Array<Matrix> = arrayOf(intMatrix(1, 0))
        val expected = floatArrayOf(2f, 1f)

        testForward(vectors, dimension, batch, false, maximumLength, inputs, expected)

    }

    @Test
    fun testFirstOfTwoVectorsOneDimension() {

        val vectors = arrayOf(floatArrayOf(1f), floatArrayOf(2f))
        val dimension = 1
        val batch = intArrayOf(0)
        val maximumLength = 2
        val inputs : Array<Matrix> = arrayOf(intMatrix(0))
        val expected = floatArrayOf(1f, Float.NaN)

        testForward(vectors, dimension, batch, false, maximumLength, inputs, expected)

    }

    @Test
    fun testLastOfTwoVectorsOneDimension() {

        val vectors = arrayOf(floatArrayOf(1f), floatArrayOf(2f))
        val dimension = 1
        val batch = intArrayOf(0)
        val maximumLength = 2
        val inputs : Array<Matrix> = arrayOf(intMatrix(1))
        val expected = floatArrayOf(2f, Float.NaN)

        testForward(vectors, dimension, batch, false, maximumLength, inputs, expected)

    }

    @Test
    fun testTwoOutOfThreeVectorsTwoDimensions() {

        val vectors = arrayOf(floatArrayOf(1f, 2f), floatArrayOf(3f, 4f), floatArrayOf(5f, 6f))
        val dimension = 2
        val batch = intArrayOf(0)
        val maximumLength = 2
        val inputs : Array<Matrix> = arrayOf(intMatrix(2, 0))
        val expected = floatArrayOf(5f, 6f, 1f, 2f)

        testForward(vectors, dimension, batch, true, maximumLength, inputs, expected)

    }

    @Test
    fun testTwoInstancesOneVectorOneDimension() {

        val vectors = arrayOf(floatArrayOf(1f))
        val dimension = 1
        val batch = intArrayOf(0, 1)
        val maximumLength = 1
        val inputs : Array<Matrix> = arrayOf(intMatrix(0), intMatrix(0))
        val expected = floatArrayOf(1f, 1f)

        testForward(vectors, dimension, batch, true, maximumLength, inputs, expected)

    }

    @Test
    fun testTwoInstancesTwoVectorsOneDimension() {

        val vectors = arrayOf(floatArrayOf(1f), floatArrayOf(2f))
        val dimension = 1
        val batch = intArrayOf(0, 1)
        val maximumLength = 1
        val inputs : Array<Matrix> = arrayOf(intMatrix(0), intMatrix(1))
        val expected = floatArrayOf(1f, 2f)

        testForward(vectors, dimension, batch, true, maximumLength, inputs, expected)

    }

    private fun testForward(vectors: Array<FloatArray>, dimension : Int, batch: IntArray, hasFixedLength : Boolean, maximumLength: Int, inputs: Array<Matrix>, expected: FloatArray) {

        val cudaContext = setUpCudaContext()

        val lookupLayer = lookupLayer(vectors, maximumLength, hasFixedLength, dimension).buildForCuda(cudaContext)

        val maximumBatchSize = batch.size

        lookupLayer.acquire(maximumBatchSize)

        val deviceResult = lookupLayer.forward(0, maximumBatchSize, batch, inputs, InputMemory())

        val actual = getFloatArray(deviceResult, batch.size * maximumLength * dimension)

        lookupLayer.release()

        assertArrayEquals(expected, actual)

        cudaContext.destroy()

    }

    @Test
    fun testBackward() {

        val cudaContext = setUpCudaContext()

        val vectors = arrayOf(floatArrayOf(1f, 2f), floatArrayOf(3f, 4f))
        val lookupLayer = lookupLayer(vectors, 3, true, 2)
            .buildForCuda(cudaContext)

        lookupLayer.acquire(1)

        lookupLayer.forward(0, 1, intArrayOf(0), arrayOf(intMatrix(0, 1, 0)), InputMemory())

        val deviceChain = Pointer()
        setFloatArray(floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f), 6, deviceChain)

        lookupLayer.backward(deviceChain)

        lookupLayer.release()

        cudaContext.destroy()

    }

}