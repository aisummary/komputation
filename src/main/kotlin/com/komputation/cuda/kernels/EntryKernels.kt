package com.komputation.cuda.kernels

object EntryKernels {

    fun lookup() = KernelInstruction(
        "lookupKernel",
        "lookupKernel",
        "entry/LookupKernel.cu",
        listOf(KernelHeaders.nan))

}