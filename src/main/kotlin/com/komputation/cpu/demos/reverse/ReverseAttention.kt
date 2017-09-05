package com.komputation.cpu.demos.reverse

import com.komputation.cpu.network.Network
import com.komputation.cpu.layers.forward.units.simpleRecurrentUnit
import com.komputation.loss.printLoss
import com.komputation.demos.reverse.ReverseData
import com.komputation.initialization.gaussianInitialization
import com.komputation.initialization.identityInitialization
import com.komputation.layers.entry.inputLayer
import com.komputation.layers.forward.activation.ActivationFunction
import com.komputation.layers.forward.decoder.attentiveDecoder
import com.komputation.layers.forward.encoder.multiOutputEncoder
import com.komputation.loss.logisticLoss
import com.komputation.matrix.IntMath
import com.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val random = Random(1)
    val seriesLength = 6
    val numberCategories = 10
    val numberExamples = IntMath.pow(10, seriesLength)
    val hiddenDimension = 10
    val numberIterations = 10
    val batchSize = 1

    val inputs = ReverseData.generateInputs(random, numberExamples, seriesLength, numberCategories)
    val targets = ReverseData.generateTargets(inputs, seriesLength, numberCategories)

    val identityInitializationStrategy = identityInitialization()
    val gaussianInitializationStrategy = gaussianInitialization(random, 0.0f, 0.001f)

    val optimizationStrategy = stochasticGradientDescent(0.001f)

    val encoderUnit = simpleRecurrentUnit(
        seriesLength,
        numberCategories,
        hiddenDimension,
        gaussianInitializationStrategy,
        identityInitializationStrategy,
        gaussianInitializationStrategy,
        ActivationFunction.ReLU,
        optimizationStrategy)

    val encoder = multiOutputEncoder(
        encoderUnit,
        seriesLength,
        numberCategories,
        hiddenDimension
    )

    val decoder = attentiveDecoder(
        seriesLength,
        hiddenDimension,
        hiddenDimension,
        ActivationFunction.Sigmoid,
        gaussianInitializationStrategy,
        gaussianInitializationStrategy,
        optimizationStrategy)

    Network(
        batchSize,
        inputLayer(seriesLength),
        encoder,
        decoder
    )
        .training(
            inputs,
            targets,
            numberIterations,
            logisticLoss(numberCategories, seriesLength),
            printLoss
        )
        .run()

}
