package com.komputation.cpu.functions

fun pad(entries : IntArray, currentLength : Int, maximumLength: Int, symbol : Int, result : IntArray) {

    for (index in 0 until currentLength) {

        result[index] = entries[index]

    }

    for (index in currentLength until maximumLength) {

        result[index] = symbol

    }

}