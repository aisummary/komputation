package com.komputation.cuda.functions

import jcuda.Pointer
import jcuda.jcublas.JCublas2.cublasCreate
import jcuda.jcublas.JCublas2.cublasDestroy
import jcuda.jcublas.cublasHandle
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import com.komputation.cuda.allocateDeviceFloatMemory
import com.komputation.cuda.getFloatArray
import com.komputation.cuda.setFloatArray

class CublasOuterProductTest {

    /*
          1 2 3
        1 1 2 3
        2 2 4 6
     */
    @Test
    fun test() {

        val cublasHandle = cublasHandle()
        cublasCreate(cublasHandle)

        val firstPointer = Pointer()
        val firstVector = floatArrayOf(1.0f, 2.0f)
        val firstDimension = firstVector.size
        setFloatArray(firstVector, firstDimension, firstPointer)

        val secondPointer = Pointer()
        val secondVector = floatArrayOf(1.0f, 2.0f, 3.0f)
        val secondDimension = secondVector.size
        setFloatArray(secondVector, secondDimension, secondPointer)

        val resultPointer = Pointer()
        val numberResultEntries = firstDimension * secondDimension
        allocateDeviceFloatMemory(resultPointer, numberResultEntries)

        cublasOuterProduct(
            cublasHandle,
            firstDimension,
            firstPointer,
            secondDimension,
            secondPointer,
            resultPointer,
            numberResultEntries
        )

        val actual = getFloatArray(resultPointer, numberResultEntries)
        val expected = floatArrayOf(1.0f, 2.0f, 2.0f, 4.0f, 3.0f, 6.0f)

        cublasDestroy(cublasHandle)

        assertArrayEquals(expected, actual)

    }

}