package com.komputation.cpu.layers.entry

import com.komputation.cpu.layers.BaseCpuEntryPoint
import com.komputation.matrix.FloatMatrix
import com.komputation.matrix.Matrix

class CpuInputLayer internal constructor(
    name : String? = null,
    numberInputRows: Int,
    numberInputColumns: Int) : BaseCpuEntryPoint(name) {

    override val numberOutputRows = numberInputRows
    override val numberOutputColumns = numberInputColumns
    override var forwardResult = FloatArray(0)

    override fun forward(input: Matrix): FloatArray {

        this.forwardResult = (input as FloatMatrix).entries

        return this.forwardResult

    }

    override fun backward(chain : FloatArray) =

        chain

}