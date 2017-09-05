package com.komputation.cpu.demos.negation

import com.komputation.cpu.network.Network
import com.komputation.loss.printLoss
import com.komputation.demos.negation.NegationData
import com.komputation.initialization.heInitialization
import com.komputation.layers.entry.inputLayer
import com.komputation.layers.forward.activation.ActivationFunction
import com.komputation.layers.forward.dense.denseLayer
import com.komputation.loss.squaredLoss
import com.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val inputDimension = 1
    val outputDimension = 1
    val batchSize = 2

    val random = Random(1)
    val initialization = heInitialization(random)

    val optimization = stochasticGradientDescent(0.01f)

    Network(
        batchSize,
        inputLayer(inputDimension),
        denseLayer(inputDimension, outputDimension, initialization, initialization, ActivationFunction.Sigmoid, optimization)
    )
        .training(
            NegationData.inputs,
            NegationData.targets,
            10_000,
            squaredLoss(outputDimension),
            printLoss)
        .run()

}