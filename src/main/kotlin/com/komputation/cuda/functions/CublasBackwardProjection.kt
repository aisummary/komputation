package com.komputation.cuda.functions

import jcuda.Pointer
import jcuda.jcublas.cublasHandle

/*
    W = w11 w12
        w21 w22
        w32 w32

    X = x1 y1
        x2 y2

    W * X =                     x1                     y1
                                x2                     y2
            w11 w12    w11 * x1 + w12 * x2    w11 * y1 + w12 * y2
            w21 w22    w21 * x1 + w22 * x2    w21 * y1 + w22 * y2
            w31 w32    w31 * x1 + w32 * x2    w31 * y1 + w32 * y2

    chain = chain11 chain12
            chain21 chain22
            chain31 chain32

    chain (.) W * X = chain11 chain12     w11 * x1 + w12 * x2, w11 * y1 + w12 * y2
                      chain21 chain22 (.) w21 * x1 + w22 * x2, w21 * y1 + w22 * y2
                      chain31 chain32     w31 * x1 + w32 * x2, w31 * y1 + w32 * y2

                    = chain11 * (w11 * x1 + w12 * x2), chain12 * (w11 * y1 + w12 * y2)
                      chain21 * (w21 * x1 + w22 * x2), chain22 * (w21 * y1 + w22 * y2)
                      chain31 * (w31 * x1 + w32 * x2), chain32 * (w31 * y1 + w32 * y2)

                    = chain11 * w11 * x1 + chain11 * w12 * x2, chain12 * w11 * y1 + chain12 * w12 * y2
                      chain21 * w21 * x1 + chain21 * w22 * x2, chain22 * w21 * y1 + chain22 * w22 * y2
                      chain31 * w31 * x1 + chain31 * w32 * x2, chain32 * w31 * y1 + chain32 * w32 * y2

*/

/*
    Differentiation w.r.t input:

    d W * x / d x1 = d (W * x)_11 / x1    d (W * x)_12 / x1
                     d (W * x)_21 / x1    d (W * x)_22 / x1

    chain (.) d W * x / d x1 = d (chain (.) W * x)_11 / x1    d (chain (.) W * x)_12 / x1
                               d (chain (.) W * x)_21 / x1    d (chain (.) W * x)_22 / x1
                               d (chain (.) W * x)_31 / x1    d (chain (.) W * x)_32 / x1
                             = chain11 * w11    0
                               chain21 * w21    0
                               chain31 * w31    0

    d chain (.) W * X / d x1 = chain11 * w11 + chain21 * w21 + chain31 * w31
    d chain (.) W * X / d x2 = chain11 * w12 + chain21 * w22 + chain31 * w32
    d chain (.) W * X / d y1 = chain12 * w11 + chain22 * w21 + chain32 * w32
    d chain (.) W * X / d y2 = chain12 * w12 + chain22 * w22 + chain32 * w32


                                                      chain11                                         chain12
                                                      chain21                                         chain22
                                                      chain31                                         chain32
    transposed W >> w11 w21 w31    chain11 * w11 + chain21 * w21 + chain31 * w31   chain12 * w11 + chain22 * w21 + chain32 * w31
                    w12 w22 w32    chain11 * w12 + chain21 * w22 + chain31 * w32   chain12 * w12 + chain22 * w22 + chain32 * w32
 */
fun cublasBackwardProjectionWrtInput(
    cublasHandle: cublasHandle,
    deviceWeights: Pointer,
    numberWeightRows: Int,
    numberWeightColumns: Int,
    deviceChain: Pointer,
    numberChainRows: Int,
    numberChainColumns: Int,
    deviceResult: Pointer) =

    // X is a vector
    if (numberChainColumns == 1) {

        cublasTransposedMatrixVectorMultiplication(cublasHandle, deviceWeights, numberWeightRows, numberWeightColumns, deviceChain, deviceResult)

    }
    // X is a matrix
    else {

        cublasTransposedMatrixMatrixMultiplication(cublasHandle, deviceWeights, numberWeightRows, numberWeightColumns, deviceChain, numberChainRows, numberChainColumns, deviceResult)

    }

/*
    chain11 * w11 * x1 + chain11 * w12 * x2, chain12 * w11 * y1 + chain12 * w12 * y2
    chain21 * w21 * x1 + chain21 * w22 * x2, chain22 * w21 * y1 + chain22 * w22 * y2
    chain31 * w31 * x1 + chain31 * w32 * x2, chain32 * w31 * y1 + chain32 * w32 * y2

    Differentiation w.r.t weights:
    d chain (.) W * X / d w11 = chain11 * x1 + chain12 * y1
    d chain (.) W * X / d w21 = chain21 * x1 + chain22 * y1
    d chain (.) W * X / d w31 = chain31 * x1 + chain32 * y1
    d chain (.) W * X / d w12 = chain11 * x2 + chain12 * y2
    d chain (.) W * X / d w22 = chain21 * x2 + chain22 * y2
    d chain (.) W * X / d w32 = chain31 * x2 + chain32 * y2

                        x1 x2  << transposed X
                        y1 y2
    chain11 chain12
    chain21 chain22
    chain31 chain32
 */
fun cublasBackwardProjectionWrtWeights(
    cublasHandle: cublasHandle,
    deviceChain: Pointer,
    numberChainRows : Int,
    numberChainColumns : Int,
    deviceInput: Pointer,
    numberInputRows : Int,
    deviceResult: Pointer,
    numberResultEntries: Int) =

    // X is a vector
    if(numberChainColumns == 1) {

        cublasOuterProduct(
            cublasHandle,
            numberChainRows,
            deviceChain,
            numberInputRows,
            deviceInput,
            deviceResult,
            numberResultEntries)

    }
    // X is a matrix
    else {

        cublasMatrixTransposedMatrixMultiplication(
            cublasHandle,
            deviceChain,
            numberChainRows,
            numberChainColumns,
            deviceInput,
            numberInputRows,
            deviceResult)

    }

fun cublasBackwardProjectionWrtBias(
    cublasHandle: cublasHandle,
    deviceChain : Pointer,
    numberChainRows : Int,
    numberChainColumns : Int,
    deviceOnes : Pointer,
    deviceResult: Pointer) {

    cublasMatrixVectorMultiplication(
        cublasHandle,
        deviceChain,
        numberChainRows,
        numberChainColumns,
        deviceOnes,
        deviceResult)

}