package com.komputation.cuda

import jcuda.Pointer
import jcuda.Sizeof
import jcuda.runtime.JCuda.*
import jcuda.runtime.cudaMemcpyKind

fun computeDeviceIntArraySize(arraySize : Int) =

    (arraySize * Sizeof.INT).toLong()

fun allocateDeviceIntMemory(pointer: Pointer, size : Int) =

    cudaMalloc(pointer, computeDeviceIntArraySize(size))

fun copyIntArrayFromDeviceToHost(source: Pointer, destination: IntArray, size: Int) =

    cudaMemcpy(Pointer.to(destination), source, computeDeviceIntArraySize(size), cudaMemcpyKind.cudaMemcpyDeviceToHost)

fun getIntArray(devicePointer: Pointer, size: Int): IntArray {

    val hostVector = IntArray(size)

    copyIntArrayFromDeviceToHost(devicePointer, hostVector, size)

    return hostVector

}

fun copyIntArrayFromHostToDevice(source: IntArray, destination: Pointer, size: Int) =

    cudaMemcpy(destination, Pointer.to(source), computeDeviceIntArraySize(size), cudaMemcpyKind.cudaMemcpyHostToDevice)


fun setIntArray(data: IntArray, size: Int, pointer : Pointer): Int {

    allocateDeviceIntMemory(pointer, size)

    return copyIntArrayFromHostToDevice(data, pointer, size)

}

fun setArrayToZero(devicePointer: Pointer, size: Int) =

    cudaMemset(devicePointer, 0, computeDeviceFloatArraySize(size))
