package com.komputation.cuda.layers.forward.activation

import jcuda.jcublas.cublasHandle
import org.junit.jupiter.api.Test
import com.komputation.cuda.CudaContext
import com.komputation.layers.forward.activation.exponentiationLayer
import com.komputation.matrix.FloatMath

class CudaExponentiationLayerTest : BaseCudaEntrywiseActivationLayerTest() {

    override fun createLayer(context: CudaContext, numberRows: Int) =

        exponentiationLayer(numberRows).buildForCuda(context, cublasHandle())

    @Test
    fun testForwardOneOfTwoInstancesOneDimensional() {

        val input = floatArrayOf(1.0f, 0.0f)
        val expected = floatArrayOf(FloatMath.exp(1.0f), Float.NaN)

        testForward(input, 1, 2, expected)

    }

    @Test
    fun testForwardOneOfTwoInstancesTwoDimensional() {

        val input = floatArrayOf(1.0f, 2.0f, Float.NaN, Float.NaN)
        val expected = floatArrayOf(FloatMath.exp(1.0f), FloatMath.exp(2.0f), Float.NaN, Float.NaN)

        testForward(input, 1, 2, expected)

    }


    @Test
    fun testBackwardOneDimension() {

        val input = floatArrayOf(1.0f, Float.NaN)
        val chain = floatArrayOf(2.0f, Float.NaN)
        val expected = floatArrayOf(2.0f * FloatMath.exp(1.0f), Float.NaN)

        testBackward(input, chain, 1, 2, expected)

    }

    @Test
    fun testBackwardTwoDimensions() {

        val input = floatArrayOf(0.0f, 1.0f)
        val chain = floatArrayOf(2.0f, 3.0f)
        val expected = floatArrayOf(2.0f, 3.0f * FloatMath.exp(1.0f))

        testBackward(input, chain, 1, 1, expected)

    }


}