package com.komputation.initialization

interface InitializationStrategy {

    fun initialize(indexRow : Int, indexColumn : Int, numberIncoming : Int) : Float

}