package com.komputation.cpu.network

import com.komputation.cpu.layers.CpuEntryPoint
import com.komputation.cpu.layers.CpuForwardLayer

class CpuBackwardPropagator(
    private val entryPoint: CpuEntryPoint,
    private val layers : Array<CpuForwardLayer>) {

    private val numberLayers = this.layers.size

    fun backward(withinBatch: Int, lossGradient: FloatArray) : FloatArray {

        var chain = lossGradient

        for(indexLayer in this.numberLayers - 1 downTo 0) {

            val layer = this.layers[indexLayer]

            chain = layer.backward(withinBatch, chain)

        }

        val result = this.entryPoint.backward(chain)

        return result

    }


}