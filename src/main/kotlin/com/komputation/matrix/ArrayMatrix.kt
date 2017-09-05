package com.komputation.matrix

sealed class Matrix {

    abstract val numberEntries : Int get
}

open class FloatMatrix(val entries: FloatArray) : Matrix() {

    override val numberEntries = entries.size

}

class IntMatrix(val entries : IntArray) : Matrix() {

    override val numberEntries = entries.size

}
