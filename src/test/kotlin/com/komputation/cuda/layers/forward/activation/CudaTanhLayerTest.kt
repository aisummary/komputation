package com.komputation.cuda.layers.forward.activation

import jcuda.jcublas.cublasHandle
import org.junit.jupiter.api.Test
import com.komputation.cuda.CudaContext
import com.komputation.layers.forward.activation.tanhLayer
import com.komputation.matrix.FloatMath

class CudaTanhLayerTest : BaseCudaEntrywiseActivationLayerTest() {

    override fun createLayer(context: CudaContext, numberRows: Int) =

        tanhLayer(numberRows).buildForCuda(context, cublasHandle())

    @Test
    fun testForwardOneOfTwoInstancesOneDimensional() {

        val input = floatArrayOf(1.0f, Float.NaN)
        val expected = floatArrayOf(FloatMath.tanh(1.0f), Float.NaN)

        testForward(input, 1, 2, expected)

    }

    @Test
    fun testForwardOneOfTwoInstancesTwoDimensional() {

        val input = floatArrayOf(1.0f, 2.0f, Float.NaN, Float.NaN)
        val expected = floatArrayOf(FloatMath.tanh(1.0f), FloatMath.tanh(2.0f), Float.NaN, Float.NaN)

        testForward(input, 1, 2, expected)

    }

    @Test
    fun testBackwardOneDimension() {

        val input = floatArrayOf(1.0f, Float.NaN)
        val chain = floatArrayOf(1.0f, Float.NaN)
        val expected = floatArrayOf(1.0f * (1.0f - FloatMath.pow(FloatMath.tanh(1.0f), 2.0f)), Float.NaN)

        testBackward(input, chain, 1, 2, expected)

    }

}