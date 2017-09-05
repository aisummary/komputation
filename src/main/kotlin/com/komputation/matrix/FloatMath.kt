package com.komputation.matrix

object FloatMath {

    fun sqrt(x : Float) =

        Math.sqrt(x.toDouble()).toFloat()

    fun pow(base: Float, exponent : Int) =

        Math.pow(base.toDouble(), exponent.toDouble()).toFloat()

    fun pow(base: Float, exponent : Float) =

        Math.pow(base.toDouble(), exponent.toDouble()).toFloat()

    fun exp(x : Float) =

        Math.exp(x.toDouble()).toFloat()

    fun log(x : Float) =

        Math.log(x.toDouble()).toFloat()

    fun tanh(x : Float) =

        Math.tanh(x.toDouble()).toFloat()


}