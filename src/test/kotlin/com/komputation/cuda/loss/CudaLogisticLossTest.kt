package com.komputation.cuda.loss

import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import com.komputation.cuda.getFloatArray
import com.komputation.cuda.setFloatArray
import com.komputation.cuda.setUpCudaContext
import com.komputation.loss.logisticLoss
import com.komputation.matrix.FloatMath

class CudaLogisticLossTest {

    @Test
    fun testForwardTwoCategoriesOneStepDimension() {

        val predictions = floatArrayOf(0.3f, 0.7f)
        val targets = floatArrayOf(0.0f, 1.0f)

        val expected = -FloatMath.log(0.7f)

        testForward(predictions, targets, 1, 1, 1, 2, expected)

    }

    @Test
    fun testForwardThreeCategoriesOneStepDimension() {

        val predictions = floatArrayOf(0.5f, 0.3f, 0.2f)
        val targets = floatArrayOf(1.0f, 0.0f, 0.0f)

        val expected = -FloatMath.log(0.5f)

        testForward(predictions, targets, 1, 1, 1, 3, expected)

    }

    @Test
    fun testForwardTwoCategoriesTwoStepsDimension() {

        val predictions = floatArrayOf(0.3f, 0.7f, 0.4f, 0.6f)
        val targets = floatArrayOf(0.0f, 1.0f, 1.0f, 0.0f)

        val expected = -FloatMath.log(0.7f) - FloatMath.log(0.4f)

        testForward(predictions, targets, 1, 1, 2, 2, expected)

    }

    @Test
    fun testBackwardTwoCategoriesOneStepDimension() {

        val predictions = floatArrayOf(0.3f, 0.7f)
        val targets = floatArrayOf(0.0f, 1.0f)

        val expected = floatArrayOf(0.0f, -1.0f/0.7f)

        testBackward(predictions, targets, 2, 1, expected)

    }

    @Test
    fun testBackwardTwoCategoriesTwoStepsDimension() {

        val predictions = floatArrayOf(0.3f, 0.7f, 0.8f, 0.2f)
        val targets = floatArrayOf(0.0f, 1.0f, 1.0f, 0.0f)

        val expected = floatArrayOf(0.0f, -1.0f/0.7f, -1.0f/0.8f, 0.0f)

        testBackward(predictions, targets, 2, 2, expected)

    }


    private fun testForward(predictions: FloatArray, targets: FloatArray, batchSize: Int, maximumBatchSize: Int, numberSteps: Int, numberCategories: Int, expected: Float) {

        val size = predictions.size

        val cudaContext = setUpCudaContext()

        val devicePredictions = Pointer()
        setFloatArray(predictions, size, devicePredictions)
        val deviceTargets = Pointer()
        setFloatArray(targets, size, deviceTargets)

        val loss = logisticLoss(numberCategories, numberSteps).buildForCuda(cudaContext)

        loss.acquire(maximumBatchSize)

        loss.accumulate(Pointer.to(devicePredictions), Pointer.to(deviceTargets), batchSize)

        val actual = loss.accessAccumulation()

        loss.release()

        cudaFree(devicePredictions)
        cudaFree(deviceTargets)

        cudaContext.destroy()

        assertEquals(expected, actual, 0.001f)

    }

    private fun testBackward(predictions: FloatArray, targets: FloatArray, numberCategories : Int, numberSteps : Int, expected: FloatArray) {

        val size = predictions.size

        val context = setUpCudaContext()

        val devicePredictions = Pointer()
        setFloatArray(predictions, size, devicePredictions)

        val deviceTargets = Pointer()
        setFloatArray(targets, size, deviceTargets)

        val loss = logisticLoss(numberCategories, numberSteps).buildForCuda(context)

        loss.acquire(1)

        val deviceResult = loss.backward(Pointer.to(devicePredictions), Pointer.to(deviceTargets), 1)
        val actual = getFloatArray(deviceResult, size)

        loss.release()

        cudaFree(devicePredictions)
        cudaFree(deviceTargets)

        context.destroy()

        assertArrayEquals(expected, actual, 0.001f)

    }


}