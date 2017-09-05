package com.komputation.cpu.demos.xor

import com.komputation.cpu.network.Network
import com.komputation.demos.xor.XorData
import com.komputation.initialization.heInitialization
import com.komputation.layers.entry.inputLayer
import com.komputation.layers.forward.activation.ActivationFunction
import com.komputation.layers.forward.dense.denseLayer
import com.komputation.loss.printLoss
import com.komputation.loss.squaredLoss
import com.komputation.optimization.historical.nesterov
import java.util.*

fun main(args: Array<String>) {

    val inputDimension = 2
    val hiddenDimension = 2
    val outputDimension = 1
    val batchSize = 4

    val random = Random(1)

    val inputLayer = inputLayer(inputDimension)

    val initialization = heInitialization(random)
    val optimization = nesterov(0.1f, 0.9f)

    val hiddenLayer = denseLayer(inputDimension, hiddenDimension, initialization, initialization, ActivationFunction.Sigmoid, optimization)
    val outputLayer = denseLayer(hiddenDimension, outputDimension, initialization, initialization, ActivationFunction.Sigmoid, optimization)

    Network(
        batchSize,
        inputLayer,
        hiddenLayer,
        outputLayer
    )
        .training(
            XorData.input,
            XorData.targets,
            10_000,
            squaredLoss(outputDimension),
            printLoss)
        .run()

}