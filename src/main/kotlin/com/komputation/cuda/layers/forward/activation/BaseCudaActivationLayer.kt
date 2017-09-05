package com.komputation.cuda.layers.forward.activation

import com.komputation.cuda.layers.BaseCudaForwardLayer

abstract class BaseCudaActivationLayer(name : String?) : BaseCudaForwardLayer(name), CudaActivationLayer