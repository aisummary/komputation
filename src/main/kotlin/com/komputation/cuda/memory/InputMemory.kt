package com.komputation.cuda.memory

import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree

class InputMemory {

    private val deviceData = hashMapOf<Int, Pointer>()
    private val deviceLengths = hashMapOf<Int, Pointer>()

    fun tryToGetData(id: Int) =

        this.deviceData[id]

    fun setData(id : Int, pointer: Pointer) {

        this.deviceData[id] = pointer

    }

    fun getLengths(id : Int) =

        this.deviceLengths[id]!!

    fun setLengths(id : Int, pointer : Pointer) {

        this.deviceLengths[id] = pointer

    }

    fun free() {

        arrayOf(this.deviceData, this.deviceLengths).forEach { map ->

            map.values.forEach { pointer ->

                cudaFree(pointer)

            }

        }

        this.deviceData.clear()
        this.deviceLengths.clear()

    }

}