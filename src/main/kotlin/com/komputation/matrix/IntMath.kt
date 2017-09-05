package com.komputation.matrix

object IntMath {

    fun ceil(x : Double) =

        Math.ceil(x).toInt()

    fun pow(base: Int, exponent: Int) =

        Math.pow(base.toDouble(), exponent.toDouble()).toInt()

    fun min(a: Int, b: Int) =

        if (a >= b) {

            b

        }
        else {

            a

        }

    fun max(a: Int, b: Int) =

        if (a >= b) {

            a

        }
        else {

            b

        }

}