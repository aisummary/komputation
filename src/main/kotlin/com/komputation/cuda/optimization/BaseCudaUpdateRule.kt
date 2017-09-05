package com.komputation.cuda.optimization

import jcuda.Pointer
import com.komputation.cuda.setIntArray
import com.komputation.layers.Resourceful
import jcuda.runtime.JCuda.cudaFree

abstract class BaseCudaUpdateRule : CudaUpdateRule, Resourceful {

    private val deviceZero = Pointer()
    private val pointerToZero = Pointer.to(this.deviceZero)

    private val deviceCountMap = hashMapOf<Int, Pointer>()

    override fun acquire(maximumBatchSize: Int) {

        setIntArray(intArrayOf(0), 1, this.deviceZero)

    }

    override fun release() {

        cudaFree(this.deviceZero)

        this.deviceCountMap.values.forEach { deviceCount ->

            cudaFree(deviceCount)

        }

    }

    override fun denseUpdate(
        count : Int,
        pointerToParameters: Pointer,
        pointerToGradient: Pointer) {

        val optionalDeviceCount = this.deviceCountMap[count]

        val deviceCount = if (optionalDeviceCount == null) {

            val deviceCount = Pointer()
            setIntArray(intArrayOf(count), 1, deviceCount)

            this.deviceCountMap[count] = deviceCount

            deviceCount
        }
        else {

            optionalDeviceCount

        }

        this.launchKernel(
            1,
            this.pointerToZero,
            Pointer.to(deviceCount),
            pointerToParameters,
            pointerToGradient)

    }

    override fun sparseUpdate(
        maximumParameters : Int,
        pointerToIndices: Pointer,
        pointerToCounts: Pointer,
        pointerToParameters: Pointer,
        pointerToGradient: Pointer) {

        this.launchKernel(
            maximumParameters,
            pointerToIndices,
            pointerToCounts,
            pointerToParameters,
            pointerToGradient)

    }

    abstract fun launchKernel(
        maximumParameters: Int,
        pointerToIndices: Pointer,
        pointerToCounts : Pointer,
        pointerToParameters: Pointer,
        pointerToGradient: Pointer) : Int

}