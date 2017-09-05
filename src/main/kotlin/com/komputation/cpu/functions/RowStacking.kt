package com.komputation.cpu.functions

fun stackRows(heights : IntArray, totalNumberRows : Int, numberColumns : Int, stacked : FloatArray, vararg arrays: FloatArray) {

    var startAtRow = 0

    for(indexArray in 0..arrays.size - 1) {

        val numberRows = heights[indexArray]

        val array = arrays[indexArray]

        for (indexColumn in 0..numberColumns - 1) {

            val startAtIndex = indexColumn * totalNumberRows

            for (indexMatrixRow in 0..numberRows - 1) {

                stacked[startAtIndex + startAtRow + indexMatrixRow] = array[indexColumn * numberRows + indexMatrixRow]

            }

        }

        startAtRow += numberRows

    }

}