package com.komputation.cpu.demos.mnist

import com.komputation.cpu.network.Network
import com.komputation.demos.mnist.MnistData
import com.komputation.initialization.constantInitialization
import com.komputation.initialization.gaussianInitialization
import com.komputation.initialization.zeroInitialization
import com.komputation.layers.entry.inputLayer
import com.komputation.layers.forward.activation.ActivationFunction
import com.komputation.layers.forward.dense.denseLayer
import com.komputation.layers.forward.highwayLayer
import com.komputation.loss.logisticLoss
import com.komputation.optimization.historical.momentum
import java.io.File
import java.util.*

fun main(args: Array<String>) {

    if (args.size != 2) {

        throw Exception("Please specify the paths to the MNIST training data and the test data (in the CSV format).")

    }

    val random = Random(1)

    val numberIterations = 10
    val batchSize = 1

    val (trainingInputs, trainingTargets) = MnistData.loadMnistTraining(File(args.first()))
    val (testInputs, testTargets) = MnistData.loadMnistTest(File(args.last()))

    val inputDimension = 784
    val hiddenDimension = 100
    val numberCategories = MnistData.numberCategories

    val gaussianInitialization = gaussianInitialization(random, 0.0f,0.0001f)
    val zeroInitialization = zeroInitialization()
    val constantInitialization = constantInitialization(-2.0f)

    val optimizer = momentum(0.01f, 0.1f)

    val dimensionalityReductionLayer = denseLayer(
        inputDimension,
        hiddenDimension,
        gaussianInitialization,
        gaussianInitialization,
        ActivationFunction.Tanh,
        optimizer
    )

    val outputLayer = denseLayer(
        hiddenDimension,
        numberCategories,
        gaussianInitialization,
        gaussianInitialization,
        ActivationFunction.Softmax,
        optimizer
    )

    val createHiddenLayer = {

        highwayLayer(hiddenDimension, gaussianInitialization, zeroInitialization, constantInitialization, ActivationFunction.Tanh, optimizer)

    }

    val network = Network(
        batchSize,
        inputLayer(inputDimension),
        dimensionalityReductionLayer,
        createHiddenLayer(),
        createHiddenLayer(),
        createHiddenLayer(),
        outputLayer
    )

    val test = network
        .test(
            testInputs,
            testTargets,
            batchSize,
            numberCategories)

    network.training(
        trainingInputs,
        trainingTargets,
        numberIterations,
        logisticLoss(numberCategories)) { _ : Int, _ : Float ->

            println(test.run())

        }
        .run()

}