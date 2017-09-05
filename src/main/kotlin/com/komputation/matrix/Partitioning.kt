package com.komputation.matrix

fun partitionIndices(size : Int, maximumBatchSize: Int): Array<IntArray> {

    val numberBatches = IntMath.ceil(size.toDouble().div(maximumBatchSize.toDouble()))

    return Array(numberBatches) { indexBatch ->

        val start = indexBatch * maximumBatchSize
        val end = Math.min(start + maximumBatchSize, size) - 1

        IntArray(end - start + 1) { index -> start + index }

    }

}