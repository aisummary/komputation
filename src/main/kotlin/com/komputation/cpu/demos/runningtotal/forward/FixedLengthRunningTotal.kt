package com.komputation.cpu.demos.runningtotal.forward

import com.komputation.cpu.layers.recurrent.Direction
import com.komputation.cpu.network.Network
import com.komputation.demos.runningtotal.RunningTotalData
import com.komputation.initialization.zeroInitialization
import com.komputation.layers.entry.inputLayer
import com.komputation.layers.forward.activation.ActivationFunction
import com.komputation.layers.recurrent.ResultExtraction
import com.komputation.layers.recurrent.recurrentLayer
import com.komputation.loss.printLoss
import com.komputation.loss.squaredLoss
import com.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val random = Random(1)

    val initialization = zeroInitialization()
    val optimization = stochasticGradientDescent(0.001f)

    val steps = 2

    val input = RunningTotalData.generateFixedLengthInput(random, steps, 0, 10, 10_000)
    val targets = RunningTotalData.generateTargets(input)

    val hasFixedLength = true

    Network(
            1,
            inputLayer(1, steps),
            recurrentLayer(steps, hasFixedLength, 1, 1, Direction.Forward, ResultExtraction.AllSteps, ActivationFunction.Identity, initialization, initialization, null, optimization)
        )
        .training(
            input,
            targets,
            2,
            squaredLoss(1, steps, hasFixedLength),
            printLoss
        )
        .run()

}