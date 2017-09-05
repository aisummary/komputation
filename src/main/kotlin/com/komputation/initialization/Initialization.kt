package com.komputation.initialization

fun initializeColumnVector(strategy: InitializationStrategy, numberRows: Int): FloatArray {

    return initializeWeights(strategy, numberRows,1, numberRows)

}

fun initializeWeights(strategy: InitializationStrategy, numberRows: Int, numberColumns : Int, numberIncoming : Int): FloatArray {

    val array = FloatArray(numberRows * numberColumns)

    for (indexRow in 0..numberRows - 1) {

        for (indexColumn in 0..numberColumns - 1) {

            array[indexColumn * numberRows + indexRow] = strategy.initialize(indexRow, indexColumn, numberIncoming)

        }

    }

    return array

}