package com.komputation.demos.xor

import com.komputation.matrix.Matrix
import com.komputation.matrix.floatMatrix

object XorData {

    val input = arrayOf<Matrix>(
        floatMatrix(0.0f, 0.0f),
        floatMatrix(1.0f, 0.0f),
        floatMatrix(0.0f, 1.0f),
        floatMatrix(1.0f, 1.0f))

    val targets = arrayOf(
        floatArrayOf(0.0f),
        floatArrayOf(1.0f),
        floatArrayOf(1.0f),
        floatArrayOf(0.0f)
    )

}