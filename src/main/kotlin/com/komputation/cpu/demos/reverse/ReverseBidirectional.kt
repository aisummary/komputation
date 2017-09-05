package com.komputation.cpu.demos.reverse

import com.komputation.cpu.network.Network
import com.komputation.cpu.layers.forward.units.simpleRecurrentUnit
import com.komputation.loss.printLoss
import com.komputation.demos.reverse.ReverseData
import com.komputation.initialization.gaussianInitialization
import com.komputation.initialization.identityInitialization
import com.komputation.initialization.zeroInitialization
import com.komputation.layers.entry.inputLayer
import com.komputation.layers.forward.activation.ActivationFunction
import com.komputation.layers.forward.concatenation
import com.komputation.layers.forward.decoder.singleInputDecoder
import com.komputation.layers.forward.encoder.singleOutputEncoder
import com.komputation.loss.logisticLoss
import com.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val random = Random(1)
    val numberExamples = 10_000
    val seriesLength = 6
    val numberCategories = 10
    val hiddenDimension = 30
    val numberIterations = 100
    val batchSize = 1

    val inputs = ReverseData.generateInputs(random, numberExamples, seriesLength, numberCategories)
    val targets = ReverseData.generateTargets(inputs, seriesLength, numberCategories)

    val identityInitialization = identityInitialization()
    val gaussianInitialization = gaussianInitialization(random, 0.0f, 0.001f)
    val zeroInitialization = zeroInitialization()

    val optimizationStrategy = stochasticGradientDescent(0.0005f)

    val forwardEncoderUnit = simpleRecurrentUnit(
        seriesLength,
        numberCategories,
        hiddenDimension,
        gaussianInitialization,
        identityInitialization,
        zeroInitialization,
        ActivationFunction.ReLU,
        optimizationStrategy
    )

    val backwardEncoderUnit = simpleRecurrentUnit(
        seriesLength,
        numberCategories,
        hiddenDimension,
        gaussianInitialization,
        identityInitialization,
        zeroInitialization,
        ActivationFunction.ReLU,
        optimizationStrategy
    )

    val decoderUnit = simpleRecurrentUnit(
        seriesLength,
        numberCategories,
        2 * hiddenDimension,
        identityInitialization,
        gaussianInitialization,
        zeroInitialization,
        ActivationFunction.ReLU,
        optimizationStrategy
    )

    Network(
        batchSize,
        inputLayer(numberCategories, seriesLength),
        concatenation(
            numberCategories,
            seriesLength,
            true,
            intArrayOf(hiddenDimension, hiddenDimension),
            1,
            arrayOf(
                singleOutputEncoder(forwardEncoderUnit, seriesLength, numberCategories, hiddenDimension, false),
                singleOutputEncoder(backwardEncoderUnit, seriesLength, numberCategories, hiddenDimension, true)
            )
        ),
        singleInputDecoder(seriesLength, 2 * hiddenDimension, numberCategories, decoderUnit, gaussianInitialization, null, ActivationFunction.Softmax, optimizationStrategy)
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
