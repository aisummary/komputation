package com.komputation.initialization

import java.util.*

class UniformInitialization internal constructor(
    private val random: Random,
    private val min: Float = 0.0f,
    max: Float = 1.0f) : InitializationStrategy {

    private val difference = max - min

    override fun initialize(indexRow: Int, indexColumn: Int, numberIncoming: Int) =

        random.nextDouble().toFloat() * this.difference - Math.abs(this.min)

}

fun uniformInitialization(random: Random, min: Float = 0.0f, max: Float = 1.0f) =

    UniformInitialization(random, min, max)