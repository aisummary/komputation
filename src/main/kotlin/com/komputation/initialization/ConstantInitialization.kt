package com.komputation.initialization

class ConstantInitialization internal constructor(private val constant: Float) : InitializationStrategy {

    override fun initialize(indexRow: Int, indexColumn: Int, numberIncoming: Int)  =

        constant
}

fun constantInitialization(constant: Float) = ConstantInitialization(constant)