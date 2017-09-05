package com.komputation.cuda.kernels.launch

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class RowwiseTest {

    val maximumNumberThreadsPerBlock = 1024

    @Test
    fun test1() {

        assertEquals(
            KernelLaunchConfiguration(1, 1, 1),
            computeRowwiseLaunchConfiguration(1, 1, maximumNumberThreadsPerBlock)
        )

    }

    @Test
    fun test2() {

        assertEquals(
            KernelLaunchConfiguration(1, 2, 1),
            computeRowwiseLaunchConfiguration(1, 2, maximumNumberThreadsPerBlock)
        )

    }

    @Test
    fun test3() {

        assertEquals(
            KernelLaunchConfiguration(1, maximumNumberThreadsPerBlock, 2),
            computeRowwiseLaunchConfiguration(1, 2048, maximumNumberThreadsPerBlock)
        )

    }

    @Test
    fun test4() {

        assertEquals(
            KernelLaunchConfiguration(1, maximumNumberThreadsPerBlock, 2),
            computeRowwiseLaunchConfiguration(1, 1500, maximumNumberThreadsPerBlock)
        )

    }


}