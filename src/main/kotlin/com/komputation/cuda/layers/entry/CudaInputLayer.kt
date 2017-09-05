package com.komputation.cuda.layers.entry

import jcuda.Pointer
import com.komputation.cpu.functions.concatenate
import com.komputation.cuda.layers.BaseCudaEntryPoint
import com.komputation.cuda.memory.InputMemory
import com.komputation.cuda.setFloatArray
import com.komputation.layers.Resourceful
import com.komputation.matrix.FloatMatrix
import com.komputation.matrix.Matrix

class CudaInputLayer internal constructor(
    name : String?,
    numberRows: Int,
    numberColumns : Int) : BaseCudaEntryPoint(name), Resourceful {

    override val hasFixedLength = true

    override val numberOutputRows = numberRows
    override val maximumOutputColumns = numberColumns
    override var deviceForwardResult = Pointer()

    private val numberEntries = numberRows * numberColumns
    private var concatenation = FloatArray(0)
    private var batchInputs = emptyArray<FloatArray>()
    private var numberBatchEntries = -1

    override fun acquire(maximumBatchSize: Int) {

        this.concatenation = FloatArray(maximumBatchSize* this.numberEntries)
        this.batchInputs = Array(maximumBatchSize) { FloatArray(0) }
        this.numberBatchEntries = maximumBatchSize * numberEntries

    }

    override fun release() {

        this.concatenation = FloatArray(0)
        this.numberBatchEntries = -1

    }

    override fun forward(
        batchId : Int,
        batchSize : Int,
        batch: IntArray,
        inputs : Array<Matrix>,
        memory: InputMemory): Pointer {

        this.deviceForwardResult = getData(batchId, batch, inputs, memory)

        return this.deviceForwardResult

    }

    private fun getData(batchId : Int, batch : IntArray, inputs: Array<Matrix>, memory: InputMemory): Pointer {

        val optionalDeviceForwardPointer = memory.tryToGetData(batchId)

        return if (optionalDeviceForwardPointer == null) {

            for ((withinBatch, id) in batch.withIndex()) {

                val input = inputs[id] as FloatMatrix
                val inputEntries = input.entries

                this.batchInputs[withinBatch] = inputEntries

                concatenate(inputEntries, withinBatch * this.numberOutputRows, this.numberOutputRows, this.concatenation)

            }

            val deviceInput = Pointer()
            setFloatArray(this.concatenation, this.numberBatchEntries, deviceInput)

            memory.setData(batchId, deviceInput)

            deviceInput

        }
        else {

            optionalDeviceForwardPointer

        }

    }

    override fun backward(chain: Pointer) =

        chain


}