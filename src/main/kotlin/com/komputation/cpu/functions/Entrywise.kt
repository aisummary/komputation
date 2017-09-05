package com.komputation.cpu.functions

fun add(a: FloatArray, b: FloatArray, result : FloatArray, numberEntries : Int) {

    for (index in 0..numberEntries - 1) {

        result[index] = a[index] + b[index]

    }

}

fun subtract(a: FloatArray, b: FloatArray, numberEntries: Int, result: FloatArray) {

    for (index in 0..numberEntries - 1) {

        result[index] = a[index] - b[index]

    }

}

fun hadamard(a: FloatArray, b: FloatArray, result : FloatArray, numberEntries : Int) {

    for (index in 0..numberEntries - 1) {

        result[index] = a[index] * b[index]

    }

}

fun negate(vector: FloatArray, result : FloatArray, numberEntries: Int) {

    for (index in 0..numberEntries - 1) {

        result[index] = -vector[index]

    }

}

fun scale(vector: FloatArray, scalar : Float, result : FloatArray, numberEntries: Int) {

    for (index in 0..numberEntries - 1) {

        result[index] = scalar * vector[index]

    }

}