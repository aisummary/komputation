package com.komputation.cpu.functions

import com.komputation.matrix.xorShift
import java.util.*

fun seed(random : Random, result: IntArray, size: Int) {

    for (index in 0..size - 1) {

        result[index] = random.nextInt()

    }

}

fun nextInteger(seeds : IntArray, offset : Int, length: Int) {

    for(index in offset..offset + length - 1) {

        seeds[index] = xorShift(seeds[index])

    }

}

fun mask(length: Int, limit: Int, offset: Int, seeds: IntArray, result: BooleanArray) {

    var offsetIndex = offset
    var index = 0

    while(index < length) {

        result[index] = seeds[offsetIndex] > limit

        offsetIndex++
        index++

    }

}

fun dropout(numberEntries: Int, input: FloatArray, mask: BooleanArray, result: FloatArray) {

    for (index in 0..numberEntries - 1) {

        result[index] = if(mask[index]) input[index] else 0.0f

    }

}

fun backwardDropout(chain : FloatArray, mask : BooleanArray, result: FloatArray, numberEntries: Int) {

    for (index in 0..numberEntries - 1) {

        result[index] = if(mask[index]) chain[index] else 0.0f

    }

}