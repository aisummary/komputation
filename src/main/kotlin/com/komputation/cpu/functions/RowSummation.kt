package com.komputation.cpu.functions

fun sumRows(numberRows: Int, numberColumns : Int, entries: FloatArray, result : FloatArray) {

    for(indexRow in 0..numberRows-1) {

        var sum = 0.0f

        for (indexColumn in 0..numberColumns - 1) {

            sum += entries[indexColumn * numberRows + indexRow]


        }

        result[indexRow] = sum

    }

}