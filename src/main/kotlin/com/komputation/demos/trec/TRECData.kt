package com.komputation.demos.trec

import java.io.File

object TRECData {

    fun readExamples(source : File) =

        source
            .readLines(Charsets.ISO_8859_1)
            .map { line ->

                val split = line.split(' ')

                val category = split.first().split(":").first()
                val text = split.drop(1).dropLast(1).map { it.toLowerCase() }

                category to text

            }
            .unzip()

}