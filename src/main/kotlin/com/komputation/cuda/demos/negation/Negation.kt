package com.komputation.cuda.demos.negation

import com.komputation.loss.printLoss
import com.komputation.cuda.network.CudaNetwork
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

    val network = CudaNetwork(
        batchSize,
        inputLayer(inputDimension),
        denseLayer(inputDimension, outputDimension, initialization, initialization, ActivationFunction.Sigmoid, optimization)
    )

    val training = network
        .training(
            NegationData.inputs,
            NegationData.targets,
            10_000,
            squaredLoss(outputDimension),
            printLoss)

    training.run()

    training.free()
    network.free()

}