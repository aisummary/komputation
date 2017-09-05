package com.komputation.cuda

import jcuda.Pointer
import jcuda.Sizeof
import jcuda.runtime.JCuda.cudaMalloc
import jcuda.runtime.JCuda.cudaMemcpy
import jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost
import jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice

fun computeDeviceFloatArraySize(arraySize : Int) =

    (arraySize * Sizeof.FLOAT).toLong()

fun allocateDeviceFloatMemory(pointer: Pointer, size : Int) =

    cudaMalloc(pointer, computeDeviceFloatArraySize(size))

fun copyFloatArrayFromHostToDevice(data: FloatArray, devicePointer: Pointer, size: Int) =

    cudaMemcpy(devicePointer, Pointer.to(data), computeDeviceFloatArraySize(size), cudaMemcpyHostToDevice)

fun copyFloatArrayFromDeviceToHost(devicePointer: Pointer, data: FloatArray, size: Int) =

    cudaMemcpy(Pointer.to(data), devicePointer, computeDeviceFloatArraySize(size), cudaMemcpyDeviceToHost)

fun getFloatArray(devicePointer: Pointer, size: Int): FloatArray {

    val hostVector = FloatArray(size)

    copyFloatArrayFromDeviceToHost(devicePointer, hostVector, size)

    return hostVector

}

fun setFloatArray(data: FloatArray, size: Int, pointer : Pointer): Int {

    allocateDeviceFloatMemory(pointer, size)

    return copyFloatArrayFromHostToDevice(data, pointer, size)

}