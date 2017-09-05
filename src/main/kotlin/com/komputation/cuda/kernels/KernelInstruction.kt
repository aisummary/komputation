package com.komputation.cuda.kernels

data class KernelInstruction(
    val name : String,
    val nameExpression: String,
    val relativePath : String,
    val relativeHeaderPaths: List<String> = emptyList())