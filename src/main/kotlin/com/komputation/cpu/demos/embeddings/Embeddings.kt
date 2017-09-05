package com.komputation.cpu.demos.embeddings

import com.komputation.cpu.network.Network
import com.komputation.loss.printLoss
import com.komputation.demos.embeddings.EmbeddingData
import com.komputation.initialization.initializeColumnVector
import com.komputation.initialization.uniformInitialization
import com.komputation.layers.entry.lookupLayer
import com.komputation.layers.forward.activation.ActivationFunction
import com.komputation.layers.forward.convolution.maxPoolingLayer
import com.komputation.layers.forward.dense.denseLayer
import com.komputation.loss.squaredLoss
import com.komputation.optimization.historical.momentum
import java.util.*

fun main(args: Array<String>) {

    val random = Random(1)

    val maximumBatchSize = 1
    val numberEmbeddings = 40
    val length = 2
    val embeddingDimension = 2

    val initializationStrategy = uniformInitialization(random, -0.05f, 0.05f)

    val initializeEmbedding = { initializeColumnVector(initializationStrategy, embeddingDimension) }
    val embeddings = Array(numberEmbeddings) { initializeEmbedding() }

    val optimizationStrategy = momentum(0.01f, 0.9f)

    val inputs = EmbeddingData.inputs
    val targets = EmbeddingData.targets
    val numberClasses = EmbeddingData.numberClasses

    val hasFixedLength = true

    Network(
        maximumBatchSize,
        lookupLayer(embeddings, length, hasFixedLength, embeddingDimension, optimizationStrategy),
        maxPoolingLayer(embeddingDimension, length),
        denseLayer(embeddingDimension, numberClasses, initializationStrategy, initializationStrategy, ActivationFunction.Softmax, optimizationStrategy)
    )
        .training(
            inputs,
            targets,
            1_000,
            squaredLoss(numberClasses),
            printLoss)
        .run()

}