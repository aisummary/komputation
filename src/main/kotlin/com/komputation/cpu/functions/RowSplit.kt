package com.komputation.cpu.functions

fun splitRows(numberRows : Int, numberColumns : Int, entries : FloatArray, heights : IntArray, numberBlocks: Int, result : Array<FloatArray>) {

    var runningHeight = 0

    for (indexBlock in 0..numberBlocks - 1) {

        val height = heights[indexBlock]
        val block = result[indexBlock]

        for (indexColumn in 0..numberColumns - 1) {

            for (indexRow in 0..height - 1) {

                block[indexColumn * height + indexRow] = entries[indexColumn * numberRows + (runningHeight + indexRow)]

            }

        }

        runningHeight += height

    }

}