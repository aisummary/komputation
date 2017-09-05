package com.komputation.cpu.functions.activation

fun relu(input: FloatArray, result : FloatArray, numberEntries : Int) {

    for(index in 0..numberEntries-1) {

        result[index] = relu(input[index])

    }

}

fun relu(entry: Float) =

    Math.max(entry, 0.0f)

// d relu(x) / d x = d max(x, 0) / d x
// = 1 if x > 0.0, 0.0 otherwise
fun backwardRelu(forwardEntries : FloatArray, chainEntries : FloatArray, result : FloatArray, numberEntries : Int) {

    for(index in 0..numberEntries-1) {

        result[index] =

            if (forwardEntries[index] > 0.0)
                chainEntries[index]
            else
                0.0f

    }

}