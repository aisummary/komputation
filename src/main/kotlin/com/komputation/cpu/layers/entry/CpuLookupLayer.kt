package com.komputation.cpu.layers.entry

import com.komputation.cpu.functions.lookup
import com.komputation.cpu.layers.BaseCpuEntryPoint
import com.komputation.cpu.optimization.SparseAccumulator
import com.komputation.cpu.optimization.UpdateRule
import com.komputation.cpu.optimization.updateSparsely
import com.komputation.layers.Resourceful
import com.komputation.matrix.IntMatrix
import com.komputation.matrix.Matrix
import com.komputation.optimization.Optimizable

class CpuLookupLayer internal constructor(
    name : String?,
    private val vectors: Array<FloatArray>,
    private val minimumLength: Int,
    private val maximumLength: Int,
    private val dimension : Int,
    private val update: UpdateRule? = null) : BaseCpuEntryPoint(name), Optimizable, Resourceful {

    override var forwardResult = FloatArray(0)
    override val numberOutputRows = this.dimension
    override var numberOutputColumns = -1

    private var inputEntries = IntArray(0)

    private val numberLengths = this.maximumLength - this.minimumLength + 1
    private val forwardResultsOverPossibleLengths = Array(this.numberLengths) { index -> FloatArray((index+this.minimumLength)*this.dimension) }

    private var gradientAccumulator: SparseAccumulator? = null

    override fun acquire(maximumBatchSize: Int) {

        this.gradientAccumulator = SparseAccumulator(this.vectors.size, maximumBatchSize, this.maximumLength, this.dimension)

    }

    override fun release() {

        this.gradientAccumulator = null

    }

    override fun forward(input: Matrix): FloatArray {

        input as IntMatrix

        this.inputEntries = input.entries
        this.numberOutputColumns = this.inputEntries.size

        this.forwardResult = this.forwardResultsOverPossibleLengths[this.numberOutputColumns - this.minimumLength]

        lookup(this.vectors, this.dimension, this.numberOutputColumns, this.inputEntries, this.forwardResult)

        return this.forwardResult

    }

    override fun backward(chain : FloatArray): FloatArray {

        this.gradientAccumulator!!.accumulate(this.inputEntries, this.numberOutputColumns, chain)

        return chain

    }

    override fun optimize(batchSize : Int) {

        if (this.update != null) {

            val gradientAccumulator = this.gradientAccumulator!!

            val size = gradientAccumulator.getSize()
            val ids = gradientAccumulator.getIds()
            val counts = gradientAccumulator.getCounts()
            val gradients = gradientAccumulator.getSums()

            updateSparsely(this.vectors, this.dimension, size, ids, counts, gradients, this.update)

        }

        this.gradientAccumulator!!.reset()

    }

}