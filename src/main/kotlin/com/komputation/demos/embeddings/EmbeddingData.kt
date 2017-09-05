package com.komputation.demos.embeddings

import com.komputation.matrix.Matrix
import com.komputation.matrix.intMatrix
import com.komputation.matrix.oneHotArray

/*

    bad
    awful
    poor
    abominable
    dreadful
    lousy
    horrible
    unpleasant
    unsatisfactory
    ghastly

    good
    excellent
    virtuous
    great
    satisfactory
    pleasant
    worthy
    beneficial
    splendid
    lovely

    moderately
    pretty
    fairly
    somewhat
    reasonably
    slightly
    mildly
    kind of
    relatively
    sort of

    extremely
    very
    extraordinarily
    exceptionally
    remarkably
    immensely
    unusually
    terribly
    totally
    uncommonly

 */

object EmbeddingData {

    private val negativeIndices = 0..9
    private val positiveIndices = 10..19
    private val weakModifierIndices = 20..29
    private val strongModifierIndices = 30..39

    private fun generateEmbeddings(modifierIndices: IntRange, polarityIndices: IntRange) =

        modifierIndices.zip(polarityIndices).map { (modifier, polarity) -> intMatrix(modifier, polarity) }

    val inputs = listOf<List<Matrix>>(
            generateEmbeddings(strongModifierIndices, negativeIndices),
            generateEmbeddings(weakModifierIndices, negativeIndices),
            generateEmbeddings(weakModifierIndices, positiveIndices),
            generateEmbeddings(strongModifierIndices, positiveIndices)
        )
        .flatten()
        .toTypedArray()

    val numberClasses = 4

    private fun generateTargets(category : Int) = (0..9).map { oneHotArray(this.numberClasses, category) }

    val targets = listOf(
            generateTargets(0),
            generateTargets(1),
            generateTargets(2),
            generateTargets(3)
        )
        .flatten()
        .toTypedArray()

}