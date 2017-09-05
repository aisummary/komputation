package com.komputation.cuda.kernels.launch

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class ColumnwiseTest {

    val maximumNumberThreadsPerBlock = 1024

    @Test
    fun test1() {

        assertEquals(
            KernelLaunchConfiguration(1, 1, 1),
            computeColumnwiseLaunchConfiguration(1, 1, maximumNumberThreadsPerBlock)
        )

    }

    @Test
    fun test2() {

        assertEquals(
            KernelLaunchConfiguration(1, 2, 1),
            computeColumnwiseLaunchConfiguration(2, 1, maximumNumberThreadsPerBlock)
        )

    }

    @Test
    fun test3() {

        assertEquals(
            KernelLaunchConfiguration(1, 33, 1),
            computeColumnwiseLaunchConfiguration(33, 1, maximumNumberThreadsPerBlock)
        )

    }

    @Test
    fun test4() {

        assertEquals(
            KernelLaunchConfiguration(2, 1, 1),
            computeColumnwiseLaunchConfiguration(1, 2, maximumNumberThreadsPerBlock)
        )

    }

    @Test
    fun test5() {

        assertEquals(
            KernelLaunchConfiguration(2, 33, 1),
            computeColumnwiseLaunchConfiguration(33, 2, maximumNumberThreadsPerBlock)
        )

    }

    @Test
    fun test6() {

        assertEquals(
            KernelLaunchConfiguration(1, maximumNumberThreadsPerBlock, 2),
            computeColumnwiseLaunchConfiguration(maximumNumberThreadsPerBlock + 1, 1, maximumNumberThreadsPerBlock)
        )

    }

    @Test
    fun test7() {

        assertEquals(
            KernelLaunchConfiguration(2, maximumNumberThreadsPerBlock, 2),
            computeColumnwiseLaunchConfiguration(maximumNumberThreadsPerBlock + 1, 2, maximumNumberThreadsPerBlock)
        )

    }


}