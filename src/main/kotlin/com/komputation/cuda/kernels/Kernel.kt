package com.komputation.cuda.kernels

import jcuda.Pointer
import jcuda.driver.CUfunction
import jcuda.nvrtc.JNvrtc.nvrtcDestroyProgram
import jcuda.nvrtc.nvrtcProgram
import java.io.File

class Kernel(
    private val computeCapabilities : Pair<Int, Int>,
    private val cuFile : File,
    private val name : String,
    private val nameExpression : String,
    private val headerFiles : Array<File>,
    private val includeNames : Array<String>) {

    private val program = nvrtcProgram()
    private val kernel = CUfunction()

    init {

        compileKernel(this.program, this.computeCapabilities, this.cuFile, this.name, arrayOf(this.nameExpression), this.headerFiles, this.includeNames)

        loadKernel(this.kernel, this.program, this.nameExpression)

    }

    fun launch(pointerToParameters: Pointer, numberBlocksInXDimension : Int, numberBlocksInYDimension : Int, numberThreadsPerBlock : Int, sharedMemoryBytes : Int) =

        launchKernel(this.kernel, pointerToParameters, numberBlocksInXDimension, numberBlocksInYDimension, numberThreadsPerBlock, sharedMemoryBytes)

    fun destroy() {

        nvrtcDestroyProgram(this.program)

    }

}