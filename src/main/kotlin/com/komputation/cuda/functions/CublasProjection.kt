package com.komputation.cuda.functions

import jcuda.Pointer
import jcuda.jcublas.JCublas2.*
import jcuda.jcublas.cublasHandle
import jcuda.jcublas.cublasOperation.CUBLAS_OP_N
import jcuda.jcublas.cublasOperation.CUBLAS_OP_T
import com.komputation.cuda.setArrayToZero

private val pointerToOne = Pointer.to(floatArrayOf(1.0f))
private val pointerToZero = Pointer.to(floatArrayOf(0.0f))

fun cublasOuterProduct(
    cublasHandle: cublasHandle,
    firstDimension: Int,
    deviceFirst: Pointer,
    secondDimension: Int,
    deviceSecond: Pointer,
    deviceResult: Pointer,
    resultDimension : Int): Int {

    setArrayToZero(deviceResult, resultDimension)

    return cublasSger(
        cublasHandle,
        firstDimension,
        secondDimension,
        pointerToOne,
        deviceFirst,
        1,
        deviceSecond,
        1,
        deviceResult,
        firstDimension
    )

}

fun cublasMatrixVectorMultiplication(
    cublasHandle: cublasHandle,
    deviceMatrix: Pointer,
    numberMatrixRows: Int,
    numberMatrixColumns: Int,
    deviceVector: Pointer,
    deviceResult: Pointer) =

    cublasSgemv(
        cublasHandle,
        CUBLAS_OP_N, // no transposition
        numberMatrixRows, // number of rows of matrix A
        numberMatrixColumns, // number of columns of matrix A
        pointerToOne, // alpha
        deviceMatrix, // weight pointer
        numberMatrixRows, // number weight rows
        deviceVector, // input pointer
        1, // storage spacing between elements of x
        pointerToZero, // beta
        deviceResult, // result pointer
        1) // storage spacing between elements of y


fun cublasTransposedMatrixVectorMultiplication(
    cublasHandle: cublasHandle,
    deviceMatrix: Pointer,
    numberMatrixRows: Int,
    numberMatrixColumns: Int,
    deviceVector: Pointer,
    deviceResult: Pointer) =

    cublasSgemv(
        cublasHandle,
        CUBLAS_OP_T,
        numberMatrixRows,
        numberMatrixColumns,
        pointerToOne,
        deviceMatrix,
        numberMatrixRows,
        deviceVector,
        1,
        pointerToZero,
        deviceResult,
        1)

fun cublasMatrixMatrixMultiplication(
    cublasHandle: cublasHandle,
    deviceA: Pointer,
    numberARows: Int,
    numberAColumns: Int,
    deviceB: Pointer,
    numberBRows: Int,
    numberBColumns: Int,
    deviceResult: Pointer) =

    cublasSgemm(
        cublasHandle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        numberARows,
        numberBColumns,
        numberAColumns,
        pointerToOne,
        deviceA,
        numberARows,
        deviceB,
        numberBRows,
        pointerToZero,
        deviceResult,
        numberARows)

fun cublasTransposedMatrixMatrixMultiplication(
    cublasHandle: cublasHandle,
    deviceA: Pointer,
    numberARows: Int,
    numberAColumns: Int,
    deviceB: Pointer,
    numberBRows: Int,
    numberBColumns: Int,
    deviceResult: Pointer) =

    cublasSgemm(
        cublasHandle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        numberAColumns, // number of rows of matrix op(A) and C
        numberBColumns, // number of columns of matrix op(B) and C
        numberARows, // number of columns of op(A) and rows of op(B)
        pointerToOne,
        deviceA,
        numberARows,
        deviceB,
        numberBRows,
        pointerToZero,
        deviceResult,
        numberAColumns)

fun cublasMatrixTransposedMatrixMultiplication(
    cublasHandle: cublasHandle,
    deviceA: Pointer,
    numberARows: Int,
    numberAColumns: Int,
    deviceB: Pointer,
    numberBRows: Int,
    deviceResult: Pointer) =

    cublasSgemm(
        cublasHandle,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        numberARows,
        numberBRows,
        numberAColumns,
        pointerToOne,
        deviceA,
        numberARows,
        deviceB,
        numberBRows,
        pointerToZero,
        deviceResult,
        numberARows)