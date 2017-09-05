package com.komputation.cuda.layers.forward.activation

import jcuda.jcublas.cublasHandle
import org.junit.jupiter.api.Test
import com.komputation.cpu.functions.activation.differentiateSigmoid
import com.komputation.cpu.functions.activation.sigmoid
import com.komputation.cuda.CudaContext
import com.komputation.layers.forward.activation.sigmoidLayer

class CudaSigmoidLayerTest : BaseCudaEntrywiseActivationLayerTest() {

    override fun createLayer(context: CudaContext, numberRows: Int)  =

        sigmoidLayer(numberRows).buildForCuda(context, cublasHandle())

    @Test
    fun testForwardOneOfTwoInstancesOneDimensional() {

        val input = floatArrayOf(1.0f, Float.NaN)
        val expected = floatArrayOf(sigmoid(1.0f), Float.NaN)

        testForward(input, 1, 2, expected)

    }

    @Test
    fun testForwardOneOfTwoInstancesTwoDimensional() {

        val input = floatArrayOf(1.0f, 2.0f, Float.NaN, Float.NaN)
        val expected = floatArrayOf(sigmoid(1.0f), sigmoid(2.0f), Float.NaN, Float.NaN)

        testForward(input, 1, 2, expected)

    }

    @Test
    fun testBackwardOneOfTwoInstanceOneDimensional() {

        val input = floatArrayOf(1.0f, Float.NaN)
        val chain = floatArrayOf(1.0f, Float.NaN)
        val expected = floatArrayOf(differentiateSigmoid(sigmoid(1.0f)), Float.NaN)

        testBackward(input, chain, 1, 2, expected)

    }


}