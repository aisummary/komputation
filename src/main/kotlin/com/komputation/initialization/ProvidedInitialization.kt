package com.komputation.initialization

class ProvidedInitialization internal constructor(private val entries: FloatArray, private val numberRows : Int) : InitializationStrategy {

    override fun initialize(indexRow: Int, indexColumn: Int, numberIncoming: Int)  =

        this.entries[indexRow + indexColumn * this.numberRows]
}

fun providedInitialization(entries : FloatArray, numberRows: Int) = ProvidedInitialization(entries, numberRows)