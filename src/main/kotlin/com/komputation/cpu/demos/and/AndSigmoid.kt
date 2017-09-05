package com.komputation.cpu.demos.and

import com.komputation.cpu.network.Network
import com.komputation.loss.printLoss
import com.komputation.demos.and.BinaryAndData
import com.komputation.initialization.heInitialization
import com.komputation.layers.entry.inputLayer
import com.komputation.layers.forward.activation.ActivationFunction
import com.komputation.layers.forward.dense.denseLayer
import com.komputation.loss.squaredLoss
import com.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val inputDimension = 2
    val outputDimension = 1
    val batchSize = 4

    val random = Random(1)
    val initialization = heInitialization(random)

    val optimization = stochasticGradientDescent(0.03f)

    Network(
        batchSize,
        inputLayer(inputDimension),
        denseLayer(inputDimension, outputDimension, initialization, initialization, ActivationFunction.Sigmoid, optimization)
    )
        .training(
            BinaryAndData.inputs,
            BinaryAndData.targets,
            10_000,
            squaredLoss(outputDimension),
            printLoss)
        .run()

}