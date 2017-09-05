package com.komputation.cuda.layers.entry

import jcuda.Pointer
import org.junit.Assert.assertArrayEquals
import org.junit.jupiter.api.Test
import com.komputation.cuda.*
import com.komputation.cuda.kernels.FillKernels
import com.komputation.cuda.kernels.HashtableKernels

class CudaGroupSumTest {

    @Test
    fun testOneRowOneColumn() {

        val numberRows = 1
        val maximumColumns = 1
        val mapping = intArrayOf(0)
        val chain = floatArrayOf(1.0f)
        val expected = floatArrayOf(1.0f, 0f)

        test(numberRows, maximumColumns, mapping, chain, expected)

    }

    @Test
    fun testOneRowTwoColumnsDifferentIndices() {

        val numberRows = 1
        val maximumColumns = 2
        val mapping = intArrayOf(0, 1)
        val chain = floatArrayOf(1f, 2f)
        val expected = floatArrayOf(1f, 2f, 0f, 0f)

        test(numberRows, maximumColumns, mapping, chain, expected)

    }

    @Test
    fun testOneRowTwoColumnsSameIndices() {

        val numberRows = 1
        val maximumColumns = 2
        val mapping = intArrayOf(0, 0)
        val chain = floatArrayOf(1f, 2f)
        val expected = floatArrayOf(3f, 0f, 0f, 0f)

        test(numberRows, maximumColumns, mapping, chain, expected)

    }

    @Test
    fun testOneRowThreeColumnsSameAndDifferentIndices() {

        val numberRows = 1
        val maximumColumns = 3
        val mapping = intArrayOf(0, 5, 0)
        val chain = floatArrayOf(1f, 2f, 3f)
        val expected = floatArrayOf(4f, 0f, 0f, 0f, 0f, 2f)

        test(numberRows, maximumColumns, mapping, chain, expected)

    }

    @Test
    fun testIncomplete() {

        val numberRows = 1
        val maximumColumns = 2
        val mapping = intArrayOf(0, -1)
        val chain = floatArrayOf(1.0f, Float.NaN)
        val expected = floatArrayOf(1.0f, 0f, 0f, 0f)

        test(numberRows, maximumColumns, mapping, chain, expected)

    }

    private fun test(numberRows : Int, maximumColumns : Int, mapping: IntArray, chain : FloatArray, expected : FloatArray) {

        val context = setUpCudaContext()

        val groupSum = CudaGroupSum(
            numberRows,
            maximumColumns,
            2 * maximumColumns,
            { context.createKernel(HashtableKernels.groupSum()) },
            { context.createKernel(FillKernels.oneFloatArray()) })

        groupSum.acquire(1)

        val deviceMapping = Pointer()
        setIntArray(mapping, mapping.size, deviceMapping)

        val deviceChain = Pointer()
        setFloatArray(chain, chain.size, deviceChain)

        groupSum.sum(Pointer.to(deviceMapping), Pointer.to(deviceChain))

        val actual = getFloatArray(groupSum.getDeviceSum(), 2 * chain.size)

        groupSum.release()

        context.destroy()

        assertArrayEquals(expected, actual, 0.01f)

    }


}