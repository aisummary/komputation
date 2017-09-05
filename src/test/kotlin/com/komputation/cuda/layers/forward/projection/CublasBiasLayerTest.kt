package com.komputation.cuda.layers.forward.projection

import jcuda.Pointer
import jcuda.jcublas.JCublas2.cublasCreate
import jcuda.jcublas.JCublas2.cublasDestroy
import jcuda.jcublas.cublasHandle
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import com.komputation.cuda.getFloatArray
import com.komputation.cuda.setFloatArray
import com.komputation.cuda.setUpCudaContext
import com.komputation.initialization.providedInitialization
import com.komputation.layers.forward.projection.biasLayer

class CublasBiasLayerTest {

    @Test
    fun testOneOutOfOneInstance() {

        /*
            1 + 1    3 + 1
            2 + 1    4 + 1
        */

        val bias = floatArrayOf(1.0f, 2.0f)

        val batchSize = 1
        val maximumBatchSize = 1

        val numberInputRows = 2
        val numberInputColumns = 2
        val input = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f)

        val expected = floatArrayOf(2.0f, 4.0f, 4.0f, 6.0f)

        test(bias, batchSize, maximumBatchSize, numberInputRows, numberInputColumns, input, expected)

    }

    @Test
    fun testOneOutOfTwoInstances() {

        /*
            1 + 1    3 + 1
            2 + 1    4 + 1
        */

        val bias = floatArrayOf(1.0f, 2.0f)

        val batchSize = 1
        val maximumBatchSize = 2

        val numberInputRows = 2
        val numberInputColumns = 2
        val input = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f)

        val expected = floatArrayOf(2.0f, 4.0f, 4.0f, 6.0f, Float.NaN, Float.NaN, Float.NaN, Float.NaN)

        test(bias, batchSize, maximumBatchSize, numberInputRows, numberInputColumns, input, expected)

    }


    private fun test(bias : FloatArray, batchSize : Int, maximumBatchSize : Int, numberInputRows: Int, numberInputColumns: Int, input : FloatArray, expected: FloatArray) {

        val cudaContext = setUpCudaContext()
        val cublasHandle = cublasHandle()

        cublasCreate(cublasHandle)

        val numberEntries = numberInputRows * numberInputColumns

        val biasLayer = biasLayer(numberInputRows, numberInputColumns, false, providedInitialization(bias, numberInputRows), null)
            .buildForCuda(cudaContext, cublasHandle)

        biasLayer.acquire(maximumBatchSize)

        val deviceInput = Pointer()
        setFloatArray(input, numberEntries, deviceInput)

        val deviceResult = biasLayer.forward(batchSize, deviceInput, true)

        val actual = getFloatArray(deviceResult, maximumBatchSize * numberEntries)

        biasLayer.release()

        cublasDestroy(cublasHandle)
        cudaContext.destroy()

        assertArrayEquals(expected, actual, 0.001f)

    }

    @Test
    fun testBackward() {

        val expected = floatArrayOf(4f, 6f)

        val cudaContext = setUpCudaContext()
        val cublasHandle = cublasHandle()

        cublasCreate(cublasHandle)

        val numberInputRows = 2
        val numberInputColumns = 2
        val bias = floatArrayOf(1.0f, 2.0f)

        val biasLayer = biasLayer(numberInputRows, numberInputColumns, true, providedInitialization(bias, numberInputRows), null).buildForCuda(cudaContext, cublasHandle)

        val maximumBatchSize = 2
        biasLayer.acquire(maximumBatchSize)

        val devieChain = Pointer()
        setFloatArray(floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f, 0f, 0f, 0f, 0f), 4, devieChain)
        val deviceResult = biasLayer.backward(1, devieChain)

        val actual = getFloatArray(deviceResult, numberInputColumns)

        biasLayer.release()

        cublasDestroy(cublasHandle)
        cudaContext.destroy()

        assertArrayEquals(expected, actual, 0.001f)

    }


}