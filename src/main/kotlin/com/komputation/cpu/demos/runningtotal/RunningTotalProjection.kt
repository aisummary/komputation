package com.komputation.cpu.demos.runningtotal

import com.komputation.cpu.network.Network
import com.komputation.cpu.layers.forward.units.simpleRecurrentUnit
import com.komputation.demos.runningtotal.RunningTotalData
import com.komputation.initialization.gaussianInitialization
import com.komputation.initialization.identityInitialization
import com.komputation.initialization.zeroInitialization
import com.komputation.layers.entry.inputLayer
import com.komputation.layers.forward.activation.ActivationFunction
import com.komputation.layers.forward.encoder.multiOutputEncoder
import com.komputation.layers.forward.projection.projectionLayer
import com.komputation.loss.printLoss
import com.komputation.loss.squaredLoss
import com.komputation.matrix.IntMath
import com.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val numberSteps = 4
    val exclusiveUpperLimit = 10
    val inputDimension = 1
    val hiddenDimension = 4
    val outputDimension = 1
    val numberExamples = IntMath.pow(exclusiveUpperLimit, numberSteps)
    val numberIterations = 30
    val batchSize = 4

    val random = Random(1)

    val identityInitialization = identityInitialization()
    val guassianInitialization = gaussianInitialization(random, 0.0f, 0.001f)
    val zeroInitialization = zeroInitialization()

    val optimizationStrategy = stochasticGradientDescent(0.001f)

    val inputs = RunningTotalData.generateInputs(random, numberExamples, numberSteps, exclusiveUpperLimit)

    val targets = RunningTotalData.generateTargets(inputs)

    val encoderUnit = simpleRecurrentUnit(
        numberSteps,
        inputDimension,
        hiddenDimension,
        guassianInitialization,
        identityInitialization,
        zeroInitialization,
        ActivationFunction.Identity,
        optimizationStrategy
    )

    Network(
        batchSize,
        inputLayer(inputDimension, numberSteps),
        multiOutputEncoder(encoderUnit, numberSteps, inputDimension, hiddenDimension),
        projectionLayer(hiddenDimension, numberSteps, false, outputDimension, guassianInitialization, guassianInitialization, optimizationStrategy)
    )
        .training(
            inputs,
            targets,
            numberIterations,
            squaredLoss(numberSteps),
            printLoss
        )
        .run()

}
