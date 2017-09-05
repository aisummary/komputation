package com.komputation.cpu.demos.lines

import com.komputation.cpu.network.Network
import com.komputation.loss.printLoss
import com.komputation.demos.lines.LinesData
import com.komputation.initialization.uniformInitialization
import com.komputation.layers.entry.inputLayer
import com.komputation.layers.forward.activation.ActivationFunction
import com.komputation.layers.forward.activation.reluLayer
import com.komputation.layers.forward.convolution.convolutionalLayer
import com.komputation.layers.forward.dense.denseLayer
import com.komputation.loss.logisticLoss
import com.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val numberRows = 3
    val numberColumns = 3

    val filterWidth = 3
    val filterHeight = 1
    val numberFilters = 6

    val outputDimension = 2

    val random = Random(1)
    val initialize = uniformInitialization(random, -0.05f, 0.05f)

    val optimization = stochasticGradientDescent(0.01f)

    val maximumBatchSize = 1

    Network(
        maximumBatchSize,
        inputLayer(numberRows, numberColumns),
        convolutionalLayer(numberRows, numberColumns, true, numberFilters, filterWidth, filterHeight, initialize, initialize, optimization),
        reluLayer(numberFilters),
        denseLayer(numberFilters, outputDimension, initialize, initialize, ActivationFunction.Softmax, optimization)
    )
        .training(
            LinesData.inputs,
            LinesData.targets,
            30_000,
            logisticLoss(outputDimension),
            printLoss)
        .run()

}