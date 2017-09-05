package com.komputation.cuda.kernels.launch

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class EntrywiseTest {

    val numberMultiProcessors = 2
    val maximumNumberThreadsPerBlock = 1024
    val warpSize = 32
    val numberResidentWarps = 64


    @Test
    fun test1() {

        assertEquals(
            KernelLaunchConfiguration(1, 32, 1),
            computeEntrywiseLaunchConfiguration(1, numberMultiProcessors, numberResidentWarps, warpSize, maximumNumberThreadsPerBlock))

    }

    @Test
    fun test2() {

        assertEquals(
            KernelLaunchConfiguration(1, 32, 1),
            computeEntrywiseLaunchConfiguration(2, numberMultiProcessors, numberResidentWarps, warpSize, maximumNumberThreadsPerBlock))

    }

    @Test
    fun test3() {

        assertEquals(
            KernelLaunchConfiguration(2, 32, 1),
            computeEntrywiseLaunchConfiguration(33, numberMultiProcessors, numberResidentWarps, warpSize, maximumNumberThreadsPerBlock))

    }

    @Test
    fun test4() {

        assertEquals(
            KernelLaunchConfiguration(2, 512, 1),
            computeEntrywiseLaunchConfiguration(1024, numberMultiProcessors, numberResidentWarps, warpSize, maximumNumberThreadsPerBlock))

    }

    @Test
    fun test5() {

        assertEquals(
            KernelLaunchConfiguration(4, maximumNumberThreadsPerBlock, 3),
            computeEntrywiseLaunchConfiguration(10_000, numberMultiProcessors, numberResidentWarps, warpSize, maximumNumberThreadsPerBlock))

    }

}