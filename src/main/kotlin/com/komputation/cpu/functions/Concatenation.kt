package com.komputation.cpu.functions

fun concatenate(segment : IntArray, start : Int, length : Int, result : IntArray) {

    System.arraycopy(segment, 0, result, start, length)

}

fun concatenate(segment : FloatArray, start : Int, length : Int, result : FloatArray) {

    System.arraycopy(segment, 0, result, start, length)

}