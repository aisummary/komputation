package com.komputation.cpu.demos.addition

import com.komputation.cpu.network.Network
import com.komputation.cpu.layers.forward.units.simpleRecurrentUnit
import com.komputation.loss.printLoss
import com.komputation.demos.addition.AdditionProblemData
import com.komputation.initialization.gaussianInitialization
import com.komputation.initialization.identityInitialization
import com.komputation.initialization.zeroInitialization
import com.komputation.layers.entry.inputLayer
import com.komputation.layers.forward.activation.ActivationFunction
import com.komputation.layers.forward.encoder.singleOutputEncoder
import com.komputation.layers.forward.projection.projectionLayer
import com.komputation.loss.squaredLoss
import com.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val random = Random(1)
    val length = 40
    val numberExamples = 10_000
    val batchSize = 1
    val inputDimension = 2
    val hiddenDimension = 20
    val outputDimension = 1
    val numberIterations = 100

    val inputs = AdditionProblemData.generateInputs(numberExamples, random, length)

    val targets = AdditionProblemData.generateTarget(inputs, length)

    val identityInitializationStrategy = identityInitialization()
    val gaussianInitializationStrategy = gaussianInitialization(random, 0.0f, 0.001f)
    val zeroInitializationStrategy = zeroInitialization()

    val optimizationStrategy = stochasticGradientDescent(0.001f)

    val encoderUnit = simpleRecurrentUnit(
        length,
        inputDimension,
        hiddenDimension,
        identityInitializationStrategy,
        gaussianInitializationStrategy,
        zeroInitializationStrategy,
        ActivationFunction.ReLU,
        optimizationStrategy
    )

    Network(
        batchSize,
        inputLayer(inputDimension),
        singleOutputEncoder(encoderUnit, length, inputDimension, hiddenDimension),
        projectionLayer(hiddenDimension, outputDimension, gaussianInitializationStrategy, gaussianInitializationStrategy, optimizationStrategy)
    )
        .training(
            inputs,
            targets,
            numberIterations,
            squaredLoss(outputDimension),
            printLoss
        )
        .run()

}