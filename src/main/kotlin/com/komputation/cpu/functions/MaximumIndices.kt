package com.komputation.cpu.functions

fun findMaxIndicesInRows(input: FloatArray, numberRows : Int, numberColumns : Int, result : IntArray) {

    for (indexRow in 0..numberRows - 1) {

        var maxValue = input[indexRow]
        var maxIndex = indexRow

        // Go through the columns in the given row
        for (indexColumn in 1..numberColumns - 1) {

            val index = indexRow + indexColumn * numberRows

            val entry = input[index]

            if (entry > maxValue) {

                maxValue = entry
                maxIndex = index

            }

        }

        result[indexRow] = maxIndex

    }

}

fun findMaxIndex(input: FloatArray, offset : Int, limit : Int): Int {

    var maxIndex = -1
    var maxValue = Float.NEGATIVE_INFINITY

    for(index in offset..offset + limit - 1) {

        val value = input[index]

        if (value > maxValue) {

            maxValue = value
            maxIndex = index

        }

    }

    return maxIndex - offset

}

fun findMaxIndices(input: FloatArray, numberIndices: Int, length: Int, result: IntArray) {

    for (step in 0..length - 1) {

        val offset = step * numberIndices

        result[step] = findMaxIndex(input, offset, numberIndices)

    }


}