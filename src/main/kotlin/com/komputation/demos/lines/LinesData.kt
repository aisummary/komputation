package com.komputation.demos.lines

import com.komputation.matrix.Matrix
import com.komputation.matrix.floatMatrixFromRows

object LinesData {

    val inputs = arrayOf<Matrix>(

        floatMatrixFromRows(
            floatArrayOf(1.0f, 1.0f, 1.0f),
            floatArrayOf(0.0f, 0.0f, 0.0f),
            floatArrayOf(0.0f, 0.0f, 0.0f)
        ),
        floatMatrixFromRows(
            floatArrayOf(0.0f, 0.0f, 0.0f),
            floatArrayOf(1.0f, 1.0f, 1.0f),
            floatArrayOf(0.0f, 0.0f, 0.0f)
        ),
        floatMatrixFromRows(
            floatArrayOf(0.0f, 0.0f, 0.0f),
            floatArrayOf(0.0f, 0.0f, 0.0f),
            floatArrayOf(1.0f, 1.0f, 1.0f)
        ),
        floatMatrixFromRows(
            floatArrayOf(1.0f, 0.0f, 0.0f),
            floatArrayOf(1.0f, 0.0f, 0.0f),
            floatArrayOf(1.0f, 0.0f, 0.0f)
        ),
        floatMatrixFromRows(
            floatArrayOf(0.0f, 1.0f, 0.0f),
            floatArrayOf(0.0f, 1.0f, 0.0f),
            floatArrayOf(0.0f, 1.0f, 0.0f)
        ),
        floatMatrixFromRows(
            floatArrayOf(0.0f, 0.0f, 1.0f),
            floatArrayOf(0.0f, 0.0f, 1.0f),
            floatArrayOf(0.0f, 0.0f, 1.0f)
        )
    )

    val targets = arrayOf(
        floatArrayOf(0.0f, 1.0f),
        floatArrayOf(0.0f, 1.0f),
        floatArrayOf(0.0f, 1.0f),
        floatArrayOf(1.0f, 0.0f),
        floatArrayOf(1.0f, 0.0f),
        floatArrayOf(1.0f, 0.0f)
    )

}