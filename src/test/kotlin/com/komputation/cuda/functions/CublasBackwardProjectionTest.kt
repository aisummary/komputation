package com.komputation.cuda.functions

import jcuda.Pointer
import jcuda.jcublas.JCublas2.cublasCreate
import jcuda.jcublas.JCublas2.cublasDestroy
import jcuda.jcublas.cublasHandle
import jcuda.runtime.JCuda.cudaFree
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import com.komputation.cuda.allocateDeviceFloatMemory
import com.komputation.cuda.getFloatArray
import com.komputation.cuda.setFloatArray
import com.komputation.cuda.setUpCudaContext
import com.komputation.matrix.FloatMatrix
import com.komputation.matrix.floatMatrix

class CublasBackwardProjectionTest {

    @Test
    fun testBackwardProjectionWrtInput1() {

        val weights = floatArrayOf(2.0f)
        val chain = floatArrayOf(3.0f)
        val expected = floatArrayOf(6.0f)

        checkBackwardProjectionWrtInput(1, 1, weights, chain, expected)

    }

    @Test
    fun testBackwardProjectionWrtInput2() {

        /*
            weights = 2
                      3
            chain = 4
                    5

                             4
                             5
            weights^T >> 2 3
         */

        val weights = floatArrayOf(2.0f, 3.0f)
        val chain = floatArrayOf(4.0f, 5.0f)
        val expected = floatArrayOf(2.0f*4.0f + 3.0f*5.0f)

        checkBackwardProjectionWrtInput(2, 1, weights, chain, expected)

    }

    @Test
    fun testBackwardProjectionWrtInput3() {

        /*
            weights = 2 3
                      4 5
            chain = 6
                    7

                              6
                              7
            weights^T >> 2 4 40
                         3 5 53
         */

        val weights = floatArrayOf(2.0f, 4.0f, 3.0f, 5.0f)
        val chain = floatArrayOf(6.0f, 7.0f)
        val expected = floatArrayOf(40.0f, 53.0f)

        checkBackwardProjectionWrtInput(2, 2, weights, chain, expected)

    }

    private fun checkBackwardProjectionWrtInput(numberWeightRows : Int, numberWeightColumns : Int, weights: FloatArray, chain: FloatArray, expected: FloatArray) {

        val context = setUpCudaContext()

        val numberWeightEntries = numberWeightRows * numberWeightColumns

        val cublasHandle = cublasHandle()
        cublasCreate(cublasHandle)

        val deviceWeights = Pointer()
        setFloatArray(weights, numberWeightEntries, deviceWeights)

        val deviceChain = Pointer()
        setFloatArray(chain, numberWeightRows, deviceChain)

        val deviceResult = Pointer()
        allocateDeviceFloatMemory(deviceResult, numberWeightColumns)

        cublasBackwardProjectionWrtInput(cublasHandle, deviceWeights, numberWeightRows, numberWeightColumns, deviceChain, numberWeightRows, 1, deviceResult)

        val hostResult = getFloatArray(deviceResult, numberWeightColumns)

        cudaFree(deviceWeights)
        cudaFree(deviceChain)
        cudaFree(deviceResult)

        cublasDestroy(cublasHandle)

        context.destroy()

        assertArrayEquals(expected, hostResult, 0.001f)

    }

    /*
    gemv solution:
            x1 x2 x3 << transposed x
    chain_1
    chain_2
    */
    @Test
    fun testBackwardProjectionWrtWeights1() {

        val input = floatMatrix(2.0f)
        val chain = floatMatrix(3.0f)
        val expected = floatArrayOf(6.0f)

        checkBackwardProjectionWrtWeights(1, 1, input, 1, 1, chain, expected)

    }

    /*
           2  3
        4  8 12
        5 10 15
     */

    @Test
    fun testBackwardProjectionWrtWeights2() {

        val input = floatMatrix(2.0f, 3.0f)
        val chain = floatMatrix(4.0f, 5.0f)
        val expected = floatArrayOf(8.0f, 10.0f, 12.0f, 15.0f)

        checkBackwardProjectionWrtWeights(2, 1, input, 2, 1, chain, expected)

    }

    private fun checkBackwardProjectionWrtWeights(
        numberInputRows : Int,
        numberInputColumns : Int,
        input: FloatMatrix,
        numberChainRows : Int,
        numberChainColumns : Int,
        chain : FloatMatrix,
        expected: FloatArray) {

        val context = setUpCudaContext()
        val inputEntries = input.entries
        val numberInputEntries = inputEntries.size

        val chainEntries = chain.entries
        val numberChainEntries = chainEntries.size

        val numberWeightEntries = numberChainRows * numberInputRows

        val cublasHandle = cublasHandle()
        cublasCreate(cublasHandle)

        val deviceInput = Pointer()
        setFloatArray(inputEntries, numberInputEntries, deviceInput)

        val deviceChain = Pointer()
        setFloatArray(chainEntries, numberChainEntries, deviceChain)

        val deviceResult = Pointer()
        allocateDeviceFloatMemory(deviceResult, numberWeightEntries)

        cublasBackwardProjectionWrtWeights(cublasHandle, deviceChain, numberChainRows, numberChainColumns, deviceInput, numberInputRows, deviceResult, numberChainRows * numberInputColumns)

        val actual = getFloatArray(deviceResult, numberWeightEntries)

        cudaFree(deviceInput)
        cudaFree(deviceChain)
        cudaFree(deviceResult)

        cublasDestroy(cublasHandle)

        context.destroy()

        assertArrayEquals(expected, actual, 0.001f)

    }

}