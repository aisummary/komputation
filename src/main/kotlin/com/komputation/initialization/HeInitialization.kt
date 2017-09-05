package com.komputation.initialization

import com.komputation.matrix.FloatMath
import java.util.*

class HeInitialization internal constructor(private val random: Random) : InitializationStrategy {

    override fun initialize(indexRow: Int, indexColumn: Int, numberIncoming: Int) =

        (random.nextGaussian().toFloat() * FloatMath.sqrt(1.0f.div(numberIncoming)))

}

fun heInitialization(random: Random) =

    HeInitialization(random)