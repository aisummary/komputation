package com.komputation.demos.mnist

import com.komputation.matrix.FloatMatrix
import com.komputation.matrix.Matrix
import com.komputation.matrix.oneHotArray
import java.io.File

object MnistData {

    val numberCategories = 10

    private fun loadMnist(csvFile: File, size: Int): Pair<Array<Matrix>, Array<FloatArray>> {

        val inputs = Array<Matrix>(size) { FloatMatrix(FloatArray(0)) }
        val targets = Array(size) { FloatArray(0) }

        csvFile
            .bufferedReader()
            .lineSequence()
            .forEachIndexed { index, line ->

                val split = line
                    .split(",")

                val category = split.first().toInt()

                val target = oneHotArray(this.numberCategories, category)

                val input = FloatMatrix(split.drop(1).map { it.toFloat().div(255.0f) }.toFloatArray())

                targets[index] = target
                inputs[index] = input

            }

        return inputs to targets

    }

    val numberTrainingExamples = 60_000
    val numberTestExamples = 10_000

    fun loadMnistTraining(csvFile: File) =

        loadMnist(csvFile, this.numberTrainingExamples)

    fun loadMnistTest(csvFile: File) =

        loadMnist(csvFile, this.numberTestExamples)

}