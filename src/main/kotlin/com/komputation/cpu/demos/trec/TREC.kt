package com.komputation.cpu.demos.trec

import com.komputation.cpu.network.Network
import com.komputation.demos.trec.NLP
import com.komputation.demos.trec.TRECData
import com.komputation.initialization.uniformInitialization
import com.komputation.layers.entry.lookupLayer
import com.komputation.layers.forward.activation.reluLayer
import com.komputation.layers.forward.activation.softmaxLayer
import com.komputation.layers.forward.convolution.convolutionalLayer
import com.komputation.layers.forward.dropout.dropoutLayer
import com.komputation.layers.forward.projection.projectionLayer
import com.komputation.loss.logisticLoss
import com.komputation.optimization.historical.nesterov
import java.io.File
import java.util.*

fun main(args: Array<String>) {

    if (args.size != 2) {

        throw Exception("Please specify the path to the Glove word embeddings and the number of dimensions.")

    }

    val embeddingFilePath = args.first()
    val dimensions = args.last().toInt()

    Trec().run(embeddingFilePath, dimensions)

}

class Trec {

    fun run(embeddingFilePath: String, embeddingDimension: Int) {

        val random = Random(1)
        val initialization = uniformInitialization(random, -0.1f, 0.1f)

        val optimization = nesterov(0.01f, 0.9f)

        val batchSize = 32
        val hasFixedLength = false
        val numberIterations = 7

        val numberFilters = 100

        val filterHeight = embeddingDimension
        val filterWidth = 2

        val keepProbability = 0.8f

        val trecDirectory = File(javaClass.classLoader.getResource("trec").toURI())
        val trainingFile = File(trecDirectory, "training.data")
        val testFile = File(trecDirectory, "test.data")

        val (trainingCategories, trainingDocuments) = TRECData.readExamples(trainingFile)
        val (testCategories, testDocuments) = TRECData.readExamples(testFile)

        val vocabulary = NLP.generateVocabulary(trainingDocuments)

        val embeddingFile = File(embeddingFilePath)

        val embeddingMap = NLP.embedVocabulary(vocabulary, embeddingFile)
        val embeddableVocabulary = embeddingMap.keys.sorted()
        val missing = vocabulary.minus(embeddingMap.keys)

        val trainingDocumentsWithFilteredTokens = NLP.filterTokens(trainingDocuments, embeddableVocabulary)
        val maximumDocumentLength = trainingDocumentsWithFilteredTokens.maxBy { document -> document.size }!!.size

        val testDocumentsWithFilteredTokens = NLP.filterTokens(testDocuments, embeddableVocabulary)

        val embeddableTrainingIndices = NLP.filterDocuments(trainingDocumentsWithFilteredTokens, filterWidth)
        val embeddableTestIndices = NLP.filterDocuments(testDocumentsWithFilteredTokens, filterWidth)

        val embeddableTrainingDocuments = trainingDocumentsWithFilteredTokens.slice(embeddableTrainingIndices)
        val embeddableTestDocuments = testDocumentsWithFilteredTokens.slice(embeddableTestIndices)

        val trainingRepresentations = NLP.vectorizeDocuments(embeddableTrainingDocuments, embeddableVocabulary)
        val testRepresentations = NLP.vectorizeDocuments(embeddableTestDocuments, embeddableVocabulary)

        val embeddableTrainingCategories = trainingCategories.slice(embeddableTrainingIndices)
        val embeddableTestCategories = testCategories.slice(embeddableTestIndices)

        val indexedCategories = NLP.indexCategories(embeddableTrainingCategories.toSet())
        val numberCategories = indexedCategories.size

        val trainingTargets = NLP.createTargets(embeddableTrainingCategories, indexedCategories)
        val testTargets = NLP.createTargets(embeddableTestCategories, indexedCategories)

        val embeddings = embeddableVocabulary
            .map { token -> embeddingMap[token]!! }
            .toTypedArray()

        val network = Network(
            batchSize,
            lookupLayer(embeddings, maximumDocumentLength, hasFixedLength, embeddingDimension, optimization),
            convolutionalLayer(embeddingDimension, maximumDocumentLength, hasFixedLength, numberFilters, filterWidth, filterHeight, initialization, initialization, optimization),
            reluLayer(numberFilters),
            dropoutLayer(random, keepProbability, numberFilters),
            projectionLayer(numberFilters, numberCategories, initialization, initialization, optimization),
            softmaxLayer(numberCategories)
        )

        val test = network
            .test(
                testRepresentations,
                testTargets,
                batchSize,
                numberCategories,
                1)

        network
            .training(
                trainingRepresentations,
                trainingTargets,
                numberIterations,
                logisticLoss(numberCategories)) { _ : Int, _: Float ->

                println(test.run())

            }
            .run()

    }

}