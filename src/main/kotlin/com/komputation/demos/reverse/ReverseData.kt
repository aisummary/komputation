package com.komputation.demos.reverse

import com.komputation.cpu.functions.getStep
import com.komputation.cpu.functions.setStep
import com.komputation.matrix.FloatMatrix
import com.komputation.matrix.Matrix
import com.komputation.matrix.oneHotArray
import java.util.*

object ReverseData {

    fun generateInputs(random : Random, numberExamples : Int, seriesLength : Int, numberCategories  : Int) =

        Array<Matrix>(numberExamples) {

            val inputEntries = FloatArray(seriesLength * numberCategories)

            for (indexStep in 0..seriesLength - 1) {

                setStep(oneHotArray(numberCategories, random.nextInt(10), 1.0f), indexStep, inputEntries, numberCategories)

            }

            FloatMatrix(inputEntries)

        }

    fun generateTargets(inputs : Array<Matrix>, seriesLength: Int, numberCategories: Int) =

        Array(inputs.size) { index ->

            val matrix = inputs[index] as FloatMatrix

            val reversedSequenceMatrix = FloatArray(numberCategories * seriesLength)

            for (indexStep in 0..seriesLength - 1) {

                val reverseStep = seriesLength - indexStep - 1

                val originalStep = FloatArray(numberCategories)
                getStep(matrix.entries, reverseStep, originalStep, numberCategories)

                setStep(originalStep, indexStep, reversedSequenceMatrix, numberCategories)
            }

            reversedSequenceMatrix

        }

}