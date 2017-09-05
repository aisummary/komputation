package com.komputation.layers.entry

import com.komputation.cpu.layers.entry.CpuInputLayer
import com.komputation.cuda.CudaContext
import com.komputation.cuda.layers.entry.CudaInputLayer
import com.komputation.layers.CpuEntryPointInstruction
import com.komputation.layers.CudaEntryPointInstruction

class InputLayer(private val name : String? = null, private val numberRows: Int, private val numberColumns: Int) : CpuEntryPointInstruction, CudaEntryPointInstruction {

    override fun buildForCpu() =

        CpuInputLayer(this.name, this.numberRows, this.numberColumns)

    override fun buildForCuda(context: CudaContext) =

        CudaInputLayer(this.name, this.numberRows, this.numberColumns)

}

fun inputLayer(numberRows : Int, numberColumns : Int = 1) =

    inputLayer(null, numberRows, numberColumns)

fun inputLayer(name : String? = null, numberRows: Int, numberColumns: Int = 1) =

    InputLayer(name, numberRows, numberColumns)