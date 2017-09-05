package com.komputation.cuda.layers.forward.dropout

import jcuda.Pointer
import jcuda.jcublas.cublasHandle
import jcuda.runtime.JCuda
import jcuda.runtime.JCuda.cudaFree
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import com.komputation.cuda.allocateDeviceFloatMemory
import com.komputation.cuda.getFloatArray
import com.komputation.cuda.setFloatArray
import com.komputation.cuda.setUpCudaContext
import com.komputation.layers.forward.dropout.dropoutLayer
import java.util.*

class CudaDropoutLayerTest {

    @Test
    fun testTraining() {

        forward(10_000, 1, 0.5f, true)

    }

    @Test
    fun testRuntime() {

        forward(10_000, 1, 0.5f, false)

    }

    private fun forward(numberRows: Int, numberColumns: Int, keepProbability: Float, isTraining : Boolean) {

        val numberEntries = numberRows * numberColumns

        val random = Random()
        val input = FloatArray(numberEntries) { random.nextFloat() }

        val cpuLayer = dropoutLayer(Random(1), keepProbability, numberRows, numberColumns).buildForCpu()

        cpuLayer.acquire(1)

        val cpuResult = cpuLayer.forward(0, 1, input, isTraining)

        val cudaContext = setUpCudaContext()

        val cudaLayer = dropoutLayer(Random(1), keepProbability, numberRows, numberColumns).buildForCuda(cudaContext, cublasHandle())
        cudaLayer.acquire(1)

        val deviceInput = Pointer()
        setFloatArray(input, numberEntries, deviceInput)

        val deviceResult = cudaLayer.forward(1, deviceInput, isTraining)
        val cudaResult = getFloatArray(deviceResult, numberEntries)

        cudaLayer.release()

        JCuda.cudaFree(deviceInput)

        cudaContext.destroy()

        assertArrayEquals(cpuResult, cudaResult, 0.001f)

    }

    @Test
    fun testBackward1() {

        val chain = floatArrayOf(1.0f, 2.0f)
        val expected = floatArrayOf(1.0f, 2.0f)

        val actual = runBackward(chain, chain.size, 1, true)

        assertArrayEquals(expected, actual, 0.001f)

    }

    @Test
    fun testBackward2() {

        val chain = floatArrayOf(1.0f, 2.0f)
        val expected = floatArrayOf(0.0f, 0.0f)

        val actual = runBackward(chain, chain.size, 1,false)

        assertArrayEquals(expected, actual, 0.001f)

    }


    private fun runBackward(chain : FloatArray, numberRows: Int, numberColumns: Int, keep : Boolean): FloatArray {

        val numberEntries = numberRows * numberColumns

        val cudaContext = setUpCudaContext()

        val cudaLayer = dropoutLayer(Random(1), if (keep) 1.0f else 0.0f, numberRows, numberColumns).buildForCuda(cudaContext, cublasHandle())

        cudaLayer.acquire(1)

        val deviceInput = Pointer()
        allocateDeviceFloatMemory(deviceInput, numberEntries)

        val deviceChain = Pointer()
        setFloatArray(chain, numberEntries, deviceChain)

        cudaLayer.forward(1, deviceInput, true)
        val deviceResult = cudaLayer.backward(1, deviceChain)

        val cudaResult = getFloatArray(deviceResult, numberEntries)

        cudaLayer.release()

        cudaFree(deviceInput)

        cudaContext.destroy()

        return cudaResult

    }

}