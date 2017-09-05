package com.komputation.cuda.memory

import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import com.komputation.cuda.setFloatArray

class TargetMemory(private val targetSize : Int) {

    private val memory = hashMapOf<Int, Pointer>()

    fun get(batchId : Int, batchSize : Int, batch : IntArray, targets : Array<FloatArray>) =

        if (this.memory.containsKey(batchId)) {

            this.memory[batchId]!!

        }
        else {

            val batchTargetSize = batchSize * this.targetSize
            val batchTargets = FloatArray(batchTargetSize)

            for ((batchIndex, globalIndex) in batch.withIndex()) {

                val target = targets[globalIndex]

                System.arraycopy(target, 0, batchTargets, batchIndex * this.targetSize, this.targetSize)

            }

            val deviceTargets = Pointer()
            setFloatArray(batchTargets, batchTargetSize, deviceTargets)

            val pointerToDeviceTargets = Pointer.to(deviceTargets)

            this.memory[batchId] = pointerToDeviceTargets

            pointerToDeviceTargets

        }

    fun free() {

        this.memory.values.forEach { pointer ->

            cudaFree(pointer)

        }

    }

}