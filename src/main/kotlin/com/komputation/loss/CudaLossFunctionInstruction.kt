package com.komputation.loss

import com.komputation.cuda.CudaContext
import com.komputation.cuda.loss.CudaLossFunction

interface CudaLossFunctionInstruction {

    fun buildForCuda(context: CudaContext) : CudaLossFunction

}