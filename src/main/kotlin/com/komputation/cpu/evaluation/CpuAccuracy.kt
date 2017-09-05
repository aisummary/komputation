package com.komputation.cpu.evaluation

fun computeAccuracy(predictions: Array<IntArray>, targets: Array<IntArray>, numberComparisons : Int): Float {

    var correctPredictions = 0

    for (indexComparison in 0..numberComparisons - 1) {

        if(targets[indexComparison].contentEquals(predictions[indexComparison])) {

            correctPredictions++

        }

    }

    return correctPredictions.toFloat().div(numberComparisons.toFloat())

}