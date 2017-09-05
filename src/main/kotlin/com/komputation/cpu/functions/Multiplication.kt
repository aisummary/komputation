package com.komputation.cpu.functions

import org.jblas.FloatMatrix
import org.jblas.SimpleBlas

fun multiply(A : FloatMatrix, B : FloatMatrix, result : FloatMatrix) {

    SimpleBlas.gemm(1.0f, A, B, 0.0f, result)

}
