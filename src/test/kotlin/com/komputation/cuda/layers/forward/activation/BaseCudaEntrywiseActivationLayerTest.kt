package com.komputation.cuda.layers.forward.activation

import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import org.junit.jupiter.api.Assertions.assertArrayEquals
import com.komputation.cuda.CudaContext
import com.komputation.cuda.getFloatArray
import com.komputation.cuda.setFloatArray
import com.komputation.cuda.setUpCudaContext

abstract class BaseCudaEntrywiseActivationLayerTest {

    protected abstract fun createLayer(context: CudaContext, numberRows: Int) : BaseCudaEntrywiseActivationLayer

    protected fun testForward(input: FloatArray, batchSize : Int, maximumBatchSize : Int, expected: FloatArray) {

        val numberEntries = input.size / maximumBatchSize

        val cudaContext = setUpCudaContext()

        val layer = createLayer(cudaContext, numberEntries)

        layer.acquire(maximumBatchSize)

        val deviceInput = Pointer()
        setFloatArray(input, numberEntries, deviceInput)

        val deviceResult = layer.forward(batchSize, deviceInput, true)
        val actual = getFloatArray(deviceResult, maximumBatchSize * numberEntries)

        cudaFree(deviceInput)

        layer.release()

        cudaContext.destroy()

        assertArrayEquals(expected, actual, 0.001f)

    }

    protected fun testBackward(input: FloatArray, chain: FloatArray, batchSize : Int, maximumBatchSize : Int, expected: FloatArray) {

        val numberEntries = input.size / maximumBatchSize

        val context = setUpCudaContext()

        val layer = createLayer(context, numberEntries)

        layer.acquire(maximumBatchSize)

        val deviceInput = Pointer()
        setFloatArray(input, numberEntries, deviceInput)
        layer.forward(batchSize, deviceInput, true)

        val deviceChain = Pointer()
        setFloatArray(chain, numberEntries, deviceChain)
        val deviceResult = layer.backward(batchSize, deviceChain)

        val actual = getFloatArray(deviceResult, maximumBatchSize * numberEntries)

        cudaFree(deviceInput)
        cudaFree(deviceChain)

        layer.release()

        assertArrayEquals(expected, actual, 0.001f)

    }


}