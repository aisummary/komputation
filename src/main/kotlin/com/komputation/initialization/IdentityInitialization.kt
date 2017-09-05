package com.komputation.initialization

class IdentityInitialization internal constructor(): InitializationStrategy {

    override fun initialize(indexRow: Int, indexColumn: Int, numberIncoming: Int) =

        if (indexRow == indexColumn) {

            1.0f

        }
        else {

            0.0f

        }

}

fun identityInitialization() = IdentityInitialization()