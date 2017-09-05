package com.komputation.cpu.layers.forward.activation

import com.komputation.cpu.layers.CpuForwardLayer
import com.komputation.layers.CpuActivationLayerInstruction
import com.komputation.layers.forward.activation.*

fun cpuActivationLayer(name: String?, function: ActivationFunction, numberRows : Int, numberColumns : Int) : CpuActivationLayerInstruction =

    when (function) {

        ActivationFunction.Identity ->
            identityLayer(name, numberRows, numberColumns)
        ActivationFunction.ReLU ->
            reluLayer(name, numberRows, numberColumns)
        ActivationFunction.Sigmoid ->
            sigmoidLayer(name, numberRows, numberColumns)
        ActivationFunction.Softmax ->
            softmaxLayer(name, numberRows, numberColumns)
        ActivationFunction.Tanh ->
            tanhLayer(name, numberRows, numberColumns)

    }

interface CpuActivationLayer : CpuForwardLayer
