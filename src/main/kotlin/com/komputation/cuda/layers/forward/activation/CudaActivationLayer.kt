package com.komputation.cuda.layers.forward.activation

import com.komputation.cuda.layers.CudaForwardLayer
import com.komputation.layers.CudaActivationLayerInstruction
import com.komputation.layers.forward.activation.*

fun cudaActivationLayer(name: String?, function: ActivationFunction, dimension : Int) : CudaActivationLayerInstruction =

    when (function) {

        ActivationFunction.Identity ->
            identityLayer(name, dimension)
        ActivationFunction.ReLU ->
            reluLayer(name, dimension)
        ActivationFunction.Sigmoid ->
            sigmoidLayer(name, dimension)
        ActivationFunction.Softmax ->
            softmaxLayer(name, dimension)
        ActivationFunction.Tanh ->
            tanhLayer(name, dimension)

    }

interface CudaActivationLayer : CudaForwardLayer