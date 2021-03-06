package com.komputation.cuda.demos.increment

import com.komputation.cpu.layers.recurrent.Direction
import com.komputation.cuda.network.cudaNetwork
import com.komputation.demos.increment.IncrementData
import com.komputation.initialization.zeroInitialization
import com.komputation.instructions.continuation.activation.RecurrentActivation
import com.komputation.instructions.entry.input
import com.komputation.instructions.loss.squaredLoss
import com.komputation.instructions.recurrent.ResultExtraction
import com.komputation.instructions.recurrent.recurrent
import com.komputation.loss.printLoss
import com.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val random = Random(1)

    val initialization = zeroInitialization()
    val optimization = stochasticGradientDescent(0.001f)

    val steps = 2

    val input = IncrementData.generateInput(random, steps, 0, 10, 10_000)
    val targets = IncrementData.generateTargets(input)

    cudaNetwork(
        1,
        input(1, steps),
        recurrent(1, RecurrentActivation.Identity, ResultExtraction.AllSteps, Direction.LeftToRight, initialization, optimization)
    )
        .training(
            input,
            targets,
            2,
            squaredLoss(),
            printLoss
        )
        .run()

}