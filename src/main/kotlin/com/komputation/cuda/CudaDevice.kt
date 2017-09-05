package com.komputation.cuda

import jcuda.driver.CUdevice
import jcuda.driver.CUdevice_attribute.*
import jcuda.driver.JCudaDriver

fun getCudaDevice(deviceId: Int): CUdevice {

    val device = CUdevice()
    JCudaDriver.cuDeviceGet(device, deviceId)

    return device
}

fun queryCudaDeviceName(device: CUdevice): String {

    val deviceName = ByteArray(1024)
    JCudaDriver.cuDeviceGetName(deviceName, deviceName.size, device)

    return buildString(deviceName)
}

fun queryDeviceAttribute(device: CUdevice, attribute : Int): Int {

    val array = intArrayOf(0)

    JCudaDriver.cuDeviceGetAttribute(array, attribute, device)

    return array.single()

}

fun queryComputeCapability(device: CUdevice) =

    queryDeviceAttribute(device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) to
    queryDeviceAttribute(device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)

fun queryNumberOfMultiprocessor(device: CUdevice) =

    queryDeviceAttribute(device, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)

fun queryMaximumNumberOfBlocks(device: CUdevice) =

    queryDeviceAttribute(device, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)

fun queryMaximumNumberOfThreadsPerBlock(device: CUdevice) =

    queryDeviceAttribute(device, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)

fun queryMaximumNumberOfResidentThreads(device: CUdevice) =

    queryDeviceAttribute(device, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)

fun queryWarpSize(device: CUdevice) =

    queryDeviceAttribute(device, CU_DEVICE_ATTRIBUTE_WARP_SIZE)

private fun buildString(bytes: ByteArray): String {

    val sb = StringBuilder()

    for (i in bytes.indices) {
        val c = bytes[i].toChar()
        if (c.toInt() == 0) {
            break
        }
        sb.append(c)
    }

    return sb.toString()

}