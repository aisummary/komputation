package com.komputation.cuda.loss

import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import com.komputation.cuda.getFloatArray
import com.komputation.cuda.setFloatArray
import com.komputation.cuda.setUpCudaContext
import com.komputation.loss.squaredLoss
import com.komputation.matrix.FloatMath

class CudaSquaredLossTest {

    @Test
    fun testForwardOneDimension() {

        val expected = 0.5f * FloatMath.pow(4.0f-2.0f, 2.0f)

        val predictions = floatArrayOf(4.0f, 0.0f)
        val targets = floatArrayOf(2.0f, 0.0f)

        testForward(predictions, targets, 1, 2, 1, expected)

    }


    @Test
    fun testForwardTwoDimensions() {

        val expected = 0.5f * (FloatMath.pow(4.0f-2.0f, 2.0f) + FloatMath.pow(6.0f-3.0f, 2.0f))

        val predictions = floatArrayOf(4.0f, 6.0f, 0.0f, 0.0f)
        val targets = floatArrayOf(2.0f, 3.0f, 0.0f, 0.0f)

        testForward(predictions, targets, 1, 2, 2, expected)

    }

    private fun testForward(predictions: FloatArray, targets: FloatArray, batchSize : Int, maximumBatchSize : Int, dimension: Int, expected: Float) {

        val size = predictions.size

        val cudaContext = setUpCudaContext()

        val devicePredictions = Pointer()
        setFloatArray(predictions, size, devicePredictions)
        val deviceTargets = Pointer()
        setFloatArray(targets, size, deviceTargets)

        val loss = squaredLoss(dimension).buildForCuda(cudaContext)

        loss.acquire(maximumBatchSize)

        loss.accumulate(Pointer.to(devicePredictions), Pointer.to(deviceTargets), batchSize)

        val actual = loss.accessAccumulation()

        loss.release()

        cudaFree(devicePredictions)
        cudaFree(deviceTargets)

        cudaContext.destroy()

        assertEquals(expected, actual, 0.001f)

    }

    @Test
    fun testBackwardOneDimension() {

        val expected = floatArrayOf(2.0f, 0.0f)

        val predictions = floatArrayOf(4.0f, 0.0f)
        val targets = floatArrayOf(2.0f, 0.0f)

        testBackward(predictions, targets, 1, 2, expected)

    }

    @Test
    fun testBackwardTwoDimensions() {

        val expected = floatArrayOf(2.0f, 3.0f)

        val predictions = floatArrayOf(4.0f, 6.0f)
        val targets = floatArrayOf(2.0f, 3.0f)

        testBackward(predictions, targets, 1, 2, expected)

    }

    private fun testBackward(predictions: FloatArray, targets: FloatArray, batchSize : Int, maximumBatchSize : Int, expected: FloatArray) {

        val size = predictions.size

        val context = setUpCudaContext()

        val devicePredictions = Pointer()
        setFloatArray(predictions, size, devicePredictions)

        val deviceTargets = Pointer()
        setFloatArray(targets, size, deviceTargets)

        val loss = squaredLoss(size).buildForCuda(context)

        loss.acquire(maximumBatchSize)

        val deviceResult = loss.backward(Pointer.to(devicePredictions), Pointer.to(deviceTargets), batchSize)
        val actual = getFloatArray(deviceResult, size)

        loss.release()

        cudaFree(devicePredictions)
        cudaFree(deviceTargets)

        context.destroy()

        assertArrayEquals(expected, actual, 0.001f)

    }


}