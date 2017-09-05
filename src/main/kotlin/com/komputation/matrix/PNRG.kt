package com.komputation.matrix

fun xorShift(seed: Int) : Int {

    var updated = seed

    updated = updated xor (updated shl 13)
    updated = updated xor (updated shr 17)
    updated = updated xor (updated shl 5)

    return updated

}