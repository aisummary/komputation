package com.komputation.cuda.kernels

import java.io.File

class KernelFactory(private val capabilities : Pair<Int, Int>) {

    private fun resolveRelativePath(relativePath: String) =

        File(this.javaClass.getResource("/cuda/$relativePath").toURI())

    fun create(instruction: KernelInstruction): Kernel {

        val kernelFile = resolveRelativePath(instruction.relativePath)

        val relativeHeaderPaths = instruction.relativeHeaderPaths
        val includeNames = Array(relativeHeaderPaths.size) { index ->

            relativeHeaderPaths[index]

        }

        val headerFiles = Array(relativeHeaderPaths.size) { index -> resolveRelativePath(relativeHeaderPaths[index]) }

        return Kernel(this.capabilities, kernelFile, instruction.name, instruction.nameExpression, headerFiles, includeNames)

    }

}