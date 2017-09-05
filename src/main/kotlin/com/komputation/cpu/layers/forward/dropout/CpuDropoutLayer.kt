package com.komputation.cpu.layers.forward.dropout

import com.komputation.cpu.functions.*
import com.komputation.cpu.layers.BaseCpuForwardLayer
import com.komputation.layers.Resourceful
import java.util.*

class CpuDropoutLayer internal constructor(
    name: String?,
    private val numberRows: Int,
    private val numberColumns: Int,
    private val random: Random,
    private val keepProbability: Float) : BaseCpuForwardLayer(name), Resourceful {

    private val numberEntries = this.numberRows * this.numberColumns

    private var entrySeeds = IntArray(this.numberEntries)

    private var mask = BooleanArray(this.numberEntries)

    override val numberOutputRows = this.numberRows
    override val numberOutputColumns = this.numberColumns
    override val forwardResult = FloatArray(this.numberEntries)

    override val numberInputRows = this.numberRows
    override val numberInputColumns = this.numberColumns
    override val backwardResult = FloatArray(this.numberEntries)

    private val dropoutProbability = 1.0 - keepProbability

    private val threshold : Int

    init {

        val numberIntegers = Math.abs(Int.MIN_VALUE.toFloat()) + Int.MAX_VALUE.toFloat()

        val numberDropoutIntegers = (this.dropoutProbability * numberIntegers).toInt()
        this.threshold = Int.MIN_VALUE + numberDropoutIntegers

    }

    override fun acquire(maximumBatchSize: Int) {

        this.entrySeeds = IntArray(maximumBatchSize * this.numberEntries)

        seed(this.random, this.entrySeeds, maximumBatchSize * this.numberEntries)

    }

    override fun release() {

        this.entrySeeds = IntArray(0)

    }

    override fun forward(withinBatch : Int, numberInputColumns: Int, input: FloatArray, isTraining: Boolean): FloatArray {

        if (isTraining) {

            val offset = withinBatch * this.numberEntries

            nextInteger(this.entrySeeds, offset, this.numberEntries)

            mask(this.numberEntries, this.threshold, offset, this.entrySeeds, this.mask)

            dropout(this.numberEntries, input, this.mask, this.forwardResult)

        }
        else {

            scale(input, this.keepProbability, this.forwardResult, this.numberEntries)

        }

        return this.forwardResult

    }

    override fun backward(withinBatch : Int, chain: FloatArray): FloatArray {

        backwardDropout(chain, this.mask, this.backwardResult, this.numberEntries)

        return this.backwardResult

    }

}