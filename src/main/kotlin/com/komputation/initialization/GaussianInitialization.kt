package com.komputation.initialization

import java.util.*

class GaussianInitialization internal constructor(
    private val random: Random, private val mean : Float = 0.0f, private val standardDeviation : Float = 1.0f) : InitializationStrategy {

    override fun initialize(indexRow: Int, indexColumn: Int, numberIncoming: Int) =

        random.nextGaussian().toFloat() * this.standardDeviation + this.mean


}

fun gaussianInitialization(random: Random, mean : Float = 0.0f, standardDeviation : Float = 1.0f) =

    GaussianInitialization(random, mean, standardDeviation)