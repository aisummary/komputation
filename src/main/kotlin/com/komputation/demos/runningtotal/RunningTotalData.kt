package com.komputation.demos.runningtotal

import com.komputation.matrix.FloatMatrix
import com.komputation.matrix.Matrix
import java.util.*

object RunningTotalData {

    fun generateInputs(random : Random, numberExamples : Int, numberSteps : Int, exclusiveUpperLimit : Int) = Array<Matrix>(numberExamples) {

        FloatMatrix(FloatArray(numberSteps) { random.nextInt(exclusiveUpperLimit).toFloat() })

    }

    fun generateTargets(inputs : Array<Matrix>) = Array<FloatArray>(inputs.size) { indexExample ->

        val input = inputs[indexExample] as FloatMatrix

        input
            .entries
            .foldIndexed(arrayListOf<Float>()) { index, list, current ->

                list.add(list.getOrElse(index-1, { 0.0f }) + current)

                list

            }
            .toFloatArray()

    }


}