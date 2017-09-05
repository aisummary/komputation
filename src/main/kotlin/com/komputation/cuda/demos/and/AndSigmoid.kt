package com.komputation.cuda.demos.and

import com.komputation.cuda.network.CudaNetwork
import com.komputation.demos.and.BinaryAndData
import com.komputation.initialization.heInitialization
import com.komputation.layers.entry.inputLayer
import com.komputation.layers.forward.activation.ActivationFunction
import com.komputation.layers.forward.dense.denseLayer
import com.komputation.loss.printLoss
import com.komputation.loss.squaredLoss
import com.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val inputDimension = 2
    val outputDimension = 1
    val maximumBatchSize = 4

    val random = Random(1)
    val initialization = heInitialization(random)

    val optimization = stochasticGradientDescent(0.03f)

    val network = CudaNetwork(
        maximumBatchSize,
        inputLayer(inputDimension),
        denseLayer(inputDimension, outputDimension, initialization, initialization, ActivationFunction.Sigmoid, optimization)
    )

    val training = network.training(BinaryAndData.inputs, BinaryAndData.targets, 10_000, squaredLoss(outputDimension), printLoss)

    training.run()

    training.free()
    network.free()

}