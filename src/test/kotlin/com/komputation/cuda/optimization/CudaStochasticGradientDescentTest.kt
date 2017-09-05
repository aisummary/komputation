package com.komputation.cuda.optimization

import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import com.komputation.cuda.getFloatArray
import com.komputation.cuda.setFloatArray
import com.komputation.cuda.setUpCudaContext
import com.komputation.optimization.stochasticGradientDescent

class CudaStochasticGradientDescentTest {

    @Test
    fun testOneDimension() {

        val cudaContext = setUpCudaContext()

        val numberRows = 1
        val numberColumns = 1
        val size = numberRows * numberColumns

        val stochasticGradientDescent = stochasticGradientDescent(0.1f).buildForCuda(cudaContext).invoke(1, numberRows, numberColumns)

        stochasticGradientDescent.acquire(1)

        val deviceParameter = Pointer()
        setFloatArray(floatArrayOf(2.0f), size, deviceParameter)
        val deviceGradient = Pointer()
        setFloatArray(floatArrayOf(0.1f), size, deviceGradient)

        stochasticGradientDescent.denseUpdate(1, Pointer.to(deviceParameter), Pointer.to(deviceGradient))

        val hostParameter = getFloatArray(deviceParameter, size)

        cudaFree(deviceParameter)
        cudaFree(deviceGradient)

        stochasticGradientDescent.release()

        cudaContext.destroy()

        assertArrayEquals(floatArrayOf(1.99f), hostParameter)

    }

}