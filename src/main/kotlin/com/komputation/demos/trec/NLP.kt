package com.komputation.demos.trec

import com.komputation.matrix.Matrix
import com.komputation.matrix.intMatrix
import com.komputation.matrix.oneHotArray
import java.io.File

object NLP {

    fun generateVocabulary(documents: Iterable<List<String>>) =

        documents
            .flatMap { tokens -> tokens }
            .toSet()

    fun embedVocabulary(vocabulary: Set<String>, embeddingFile: File): Map<String, FloatArray> {

        val embeddingMap = hashMapOf<String, FloatArray>()

        embeddingFile.bufferedReader().use { reader ->

            reader
                .lineSequence()
                .forEach { line ->

                    val split = line.split(" ")

                    val word = split.first()

                    if (vocabulary.contains(word)) {

                        val embedding = split.drop(1).map { it.toFloat() }.toFloatArray()

                        embeddingMap.put(word, embedding)

                    }

                }

            embeddingMap

        }

        return embeddingMap

    }

    fun filterTokens(documents: Iterable<List<String>>, vocabulary: Collection<String>) =

        documents
            .map { document -> document.filter { vocabulary.contains(it) } }

    fun filterDocuments(documents: Iterable<List<String>>, minLength : Int) =

        documents
            .withIndex()
            .filter { (_, document)-> document.size >= minLength }
            .map { (index, _) -> index }

    fun vectorizeDocuments(documents: Iterable<List<String>>, vocabulary: Collection<String>) =

        documents
            .map { tokens -> tokens.map { vocabulary.indexOf(it) }.toIntArray() }
            .map { indices -> intMatrix(*indices) as Matrix }
            .toTypedArray()

    fun indexCategories(categories: Set<String>) =

        categories
            .toSet()
            .sorted()
            .mapIndexed { index, category -> category to index }
            .toMap()

    fun createTargets(categories: Iterable<String>, indexedCategories: Map<String, Int>) =

        categories
            .map { category -> oneHotArray(indexedCategories.size, indexedCategories[category]!!) }
            .toTypedArray()

}