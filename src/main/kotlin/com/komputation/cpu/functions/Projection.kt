package com.komputation.cpu.functions

fun addBias(input: FloatArray, numberInputRows: Int, numberInputColumns: Int, bias: FloatArray, result: FloatArray) {

    for(indexColumn in 0..numberInputColumns - 1) {

        val firstColumnIndex = indexColumn * numberInputRows

        for (indexRow in 0..numberInputRows-1) {

            val indexEntry = firstColumnIndex + indexRow
            result[indexEntry] = input[indexEntry] + bias[indexRow]

        }

    }

}

fun backwardProjectionWrtInput(
    numberInputRows: Int,
    numberInputColumns : Int,
    weightEntries : FloatArray,
    numberWeightRows : Int,
    chainEntries: FloatArray,
    numberChainRows : Int,
    derivatives : FloatArray) {

    var index = 0

    for (indexInputColumn in 0..numberInputColumns - 1) {

        for (indexInputRow in 0..numberInputRows - 1) {

            var derivative = 0.0f

            for (indexWeightRow in 0..numberWeightRows - 1) {

                val chainEntry = chainEntries[indexWeightRow + indexInputColumn * numberChainRows]
                val weightEntry = weightEntries[indexWeightRow + indexInputRow * numberWeightRows]

                derivative += chainEntry * weightEntry

            }

            derivatives[index++] = derivative

        }

    }

}

fun backwardProjectionWrtWeights(
    numberWeightRows : Int,
    numberWeightColumns: Int,
    inputEntries: FloatArray,
    numberInputRows : Int,
    chainEntries: FloatArray,
    numberChainRows: Int,
    numberChainColumns : Int,
    result : FloatArray) {

    var index = 0

    for (indexWeightColumn in 0..numberWeightColumns - 1) {

        for (indexWeightRow in 0..numberWeightRows - 1) {

            var derivative = 0.0f

            for (indexChainColumn in 0..numberChainColumns - 1) {

                // d pre ij / d wk
                val inputEntry = inputEntries[indexWeightColumn + indexChainColumn * numberInputRows]

                if (inputEntry.isNaN()) {

                    break

                }

                // d loss / d pre1, d loss / d pre2
                // All multiplications on other rows equal to zero
                val chainEntry = chainEntries[indexWeightRow + indexChainColumn * numberChainRows]

                derivative += inputEntry * chainEntry

            }

            result[index++] = derivative

        }

    }

}

fun backwardProjectionWrtBias(numberBiasRows : Int, chain: FloatArray, numberChainRows: Int, numberChainColumns: Int, result : FloatArray) {

    for (indexRow in 0..numberBiasRows - 1) {

        var derivative = 0.0f

        for (indexChainColumn in 0..numberChainColumns - 1) {

            val chainEntry = chain[indexChainColumn * numberChainRows + indexRow]
            derivative += chainEntry

        }

        result[indexRow] = derivative

    }

}