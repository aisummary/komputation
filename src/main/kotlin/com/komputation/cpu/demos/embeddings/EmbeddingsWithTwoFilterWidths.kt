package com.komputation.cpu.demos.embeddings

import com.komputation.cpu.network.Network
import com.komputation.loss.printLoss
import com.komputation.demos.embeddings.EmbeddingData
import com.komputation.initialization.initializeColumnVector
import com.komputation.initialization.uniformInitialization
import com.komputation.layers.entry.lookupLayer
import com.komputation.layers.forward.activation.ActivationFunction
import com.komputation.layers.forward.activation.reluLayer
import com.komputation.layers.forward.concatenation
import com.komputation.layers.forward.convolution.convolutionalLayer
import com.komputation.layers.forward.dense.denseLayer
import com.komputation.loss.squaredLoss
import com.komputation.optimization.historical.momentum
import java.util.*

fun main(args: Array<String>) {

    val random = Random(1)

    val maximumBatchSize = 1
    val numberEmbeddings = 40
    val embeddingDimension = 2

    val initializationStrategy = uniformInitialization(random, -0.05f, 0.05f)

    val initializeEmbedding = { initializeColumnVector(initializationStrategy, embeddingDimension) }
    val embeddings = Array(numberEmbeddings) { initializeEmbedding() }

    val optimizationStrategy = momentum(0.01f, 0.9f)

    val numberFilters = 2
    val filterWidths = arrayOf(1, 2)
    val numberFilterWidths = filterWidths.size
    val filterHeight = embeddingDimension
    val totalNumberFilters = numberFilterWidths * numberFilters

    val numberClasses = EmbeddingData.numberClasses
    val input = EmbeddingData.inputs
    val targets = EmbeddingData.targets

    val hasFixedLength = true

    Network(
        maximumBatchSize,
        lookupLayer(embeddings, 2, hasFixedLength, embeddingDimension, optimizationStrategy),
        concatenation(
            embeddingDimension,
            2,
            hasFixedLength,
            intArrayOf(numberFilters, numberFilters),
            1,
            filterWidths
                .map { filterWidth -> convolutionalLayer(embeddingDimension, 2, hasFixedLength, numberFilters, filterWidth, filterHeight, initializationStrategy, initializationStrategy, optimizationStrategy) }
                .toTypedArray()
        ),
        reluLayer(totalNumberFilters),
        denseLayer(totalNumberFilters, numberClasses, initializationStrategy, initializationStrategy, ActivationFunction.Softmax, optimizationStrategy)
    )
        .training(
            input,
            targets,
            10_000,
            squaredLoss(numberClasses),
            printLoss)
        .run()

}