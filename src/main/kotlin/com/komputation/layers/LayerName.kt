package com.komputation.layers

fun concatenateNames(baseName: String?, appendix: String) =

    if(baseName != null)
        "$baseName-$appendix"
    else
        null