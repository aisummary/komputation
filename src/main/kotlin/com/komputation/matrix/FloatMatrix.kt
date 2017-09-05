package com.komputation.matrix

fun floatMatrix(vararg entries : Float) =

    FloatMatrix(entries)

fun floatArrayFromRows(vararg rows: FloatArray): FloatArray {

    val numberRows = rows.size
    val numberColumns = rows[0].size

    val entries = FloatArray(numberRows * numberColumns)

    for (indexRow in 0..numberRows - 1) {

        val row = rows[indexRow]

        for (indexColumn in 0..numberColumns - 1) {

            entries[indexRow + indexColumn * numberRows] = row[indexColumn]

        }

    }

    return entries

}

fun floatMatrixFromRows(vararg rows: FloatArray) =

    FloatMatrix(floatArrayFromRows(*rows))

fun oneHotArray(size: Int, index: Int, value : Float = 1.0f): FloatArray {

    val array = FloatArray(size)
    array[index] = value

    return array

}