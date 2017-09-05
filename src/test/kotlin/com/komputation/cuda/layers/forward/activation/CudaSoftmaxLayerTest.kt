package com.komputation.cuda.layers.forward.activation

import jcuda.Pointer
import jcuda.jcublas.cublasHandle
import jcuda.runtime.JCuda.cudaFree
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import com.komputation.cpu.layers.CpuForwardLayer
import com.komputation.cuda.getFloatArray
import com.komputation.cuda.layers.CudaForwardLayer
import com.komputation.cuda.setFloatArray
import com.komputation.cuda.setUpCudaContext
import com.komputation.layers.acquireRecursively
import com.komputation.layers.forward.activation.softmaxLayer
import com.komputation.layers.releaseRecursively

class CudaSoftmaxLayerTest {

    @Test
    fun testForwardOneRowOneColumn() {

        val input = floatArrayOf(0.0f)
        val numberRows = 1
        val numberColumns = 1

        testForward(numberRows, numberColumns, input, floatArrayOf(1.0f))

    }

    @Test
    fun testForwardTwoRowsOneColumn1() {

        val input = floatArrayOf(0.0f, 0.0f)
        val numberRows = 2
        val numberColumns = 1

        val expected = floatArrayOf(0.5f, 0.5f)

        testForward(numberRows, numberColumns, input, expected)

    }

    @Test
    fun testForwardTwoRowsOneColumn2() {

        val input = floatArrayOf(0.0f, 1.0f)
        val numberRows = 2
        val numberColumns = 1

        val expected = floatArrayOf(0.268941421f, 0.731058579f)

        testForward(numberRows, numberColumns, input, expected)

    }

    private fun testForward(numberRows: Int, numberColumns: Int, input: FloatArray, expected: FloatArray) {

        val actual = forward(numberRows, numberColumns, input)

        assertArrayEquals(expected, actual, 0.001f)

    }

    private fun forward(numberRows: Int, numberColumns: Int, input: FloatArray): FloatArray {

        val batchSize = 1

        val numberEntries = numberRows * numberColumns

        val cudaContext = setUpCudaContext()

        val softmaxLayer = softmaxLayer(numberRows, numberColumns).buildForCuda(cudaContext, cublasHandle())

        acquireRecursively(softmaxLayer, batchSize)

        val deviceInput = Pointer()
        setFloatArray(input, numberEntries, deviceInput)

        val deviceResult = softmaxLayer.forward(batchSize, deviceInput, true)
        val actual = getFloatArray(deviceResult, numberEntries)

        cudaFree(deviceInput)

        releaseRecursively(softmaxLayer, CudaForwardLayer::class.java)

        cudaContext.destroy()

        return actual

    }

    @Test
    fun testBackwardOneRowOneColumn() {

        val input = floatArrayOf(1.0f)
        val chain = floatArrayOf(1.0f)
        val numberRows = 1
        val numberColumns = 1

        testBackward(numberRows, numberColumns, input, chain)

    }

    @Test
    fun testBackwardTwoRowsOneColumn1() {

        val input = floatArrayOf(1.0f, 1.0f)
        val chain = floatArrayOf(1.0f, 1.0f)
        val numberRows = 2
        val numberColumns = 1

        testBackward(numberRows, numberColumns, input, chain)

    }

    @Test
    fun testBackwardTwoRowsOneColumn2() {

        val input = floatArrayOf(0.0f, 1.0f)
        val chain = floatArrayOf(1.0f, 1.0f)
        val numberRows = 2
        val numberColumns = 1

        testBackward(numberRows, numberColumns, input, chain)

    }

    @Test
    fun testBackwardOneRowTwoColumns() {

        val input = floatArrayOf(1.0f, 2.0f)
        val chain = floatArrayOf(1.0f, 2.0f)
        val numberRows = 1
        val numberColumns = 2

        testBackward(numberRows, numberColumns, input, chain)

    }

    private fun testBackward(numberRows: Int, numberColumns: Int, input: FloatArray, chain : FloatArray) {

        val numberEntries = numberRows * numberColumns

        val cudaContext = setUpCudaContext()

        val softmaxLayer = softmaxLayer(numberRows, numberColumns)

        val cpuSoftmaxLayer = softmaxLayer.buildForCpu()

        acquireRecursively(cpuSoftmaxLayer, 1)

        cpuSoftmaxLayer.forward(0, numberColumns, input, true)
        cpuSoftmaxLayer.backward(0, chain)
        val expected = cpuSoftmaxLayer.backwardResult

        val cudaSoftmaxLayer = softmaxLayer.buildForCuda(cudaContext, cublasHandle())

        acquireRecursively(cudaSoftmaxLayer, 1)

        val deviceInput = Pointer()
        setFloatArray(input, numberEntries, deviceInput)

        val deviceChain = Pointer()
        setFloatArray(chain, numberEntries, deviceChain)

        cudaSoftmaxLayer.forward(1, deviceInput, true)
        val deviceBackwardResult = cudaSoftmaxLayer.backward(1, deviceChain)
        val actual = getFloatArray(deviceBackwardResult, numberEntries)

        cudaFree(deviceInput)
        cudaFree(deviceChain)

        releaseRecursively(cpuSoftmaxLayer, CpuForwardLayer::class.java)
        releaseRecursively(cudaSoftmaxLayer, CudaForwardLayer::class.java)

        cudaContext.destroy()

        assertArrayEquals(expected, actual, 0.001f)

    }


}