package com.komputation.demos.addition

import com.komputation.cpu.functions.getStep
import com.komputation.matrix.FloatMatrix
import com.komputation.matrix.Matrix
import java.util.*

object AdditionProblemData {

    fun generateInputs(size: Int, random: Random, length: Int) =

        Array<Matrix>(size) {

            val input = FloatArray(length * 2)
            for (index in 0..length - 1) {

                input[index * 2] = random.nextFloat()

            }

            val firstStep = random.nextInt(length)
            val secondStep = random.nextInt(length).let { candidate ->

                if (candidate == firstStep) {

                    if (firstStep == length - 1) {

                        firstStep - 1

                    }
                    else {

                        firstStep + 1

                    }

                }
                else {

                    candidate

                }

            }

            input[firstStep * 2 + 1] = 1.0f
            input[secondStep * 2 + 1] = 1.0f

            FloatMatrix(input)

        }

    fun generateTarget(inputs: Array<Matrix>, numberSteps : Int) =

        Array(inputs.size) { indexInput ->

            val input = inputs[indexInput] as FloatMatrix

            var solution = 0.0f

            val inputEntries = input.entries

            for (indexStep in 0..numberSteps - 1) {

                val step = FloatArray(2)
                getStep(inputEntries, indexStep, step, 2)

                solution += step[0] * step[1]

            }

            floatArrayOf(solution)

        }
}