package com.komputation.layers

import java.lang.reflect.Array

interface Resourceful {

    fun acquire(maximumBatchSize : Int)

    fun release()

}

private val acquisitionMethod =

    Resourceful::class.java.getMethod("acquire", Integer.TYPE)

private val releaseMethod =

    Resourceful::class.java.getMethod("release")

fun acquireRecursively(obj: Any, maximumBatchSize: Int) {

    if(obj is Resourceful) {

        acquisitionMethod.invoke(obj, maximumBatchSize)

    }

    val members = findMembers(obj)

    members.forEach { member ->

        acquireRecursively(member, maximumBatchSize)

    }

}

fun releaseRecursively(obj: Any, assignableFrom : Class<*>) {

    if(obj is Resourceful) {

        releaseMethod.invoke(obj)

    }

    findMembers(obj).forEach { member ->

        releaseRecursively(member, assignableFrom)

    }

}

private fun findMembers(container: Any) =

    container
        .javaClass
        .declaredFields
        .map { field ->

            val fieldIsInaccessible = !field.isAccessible

            if (fieldIsInaccessible) {

                field.isAccessible = true

            }

            val member = field.get(container)

            if (fieldIsInaccessible) {

                field.isAccessible = false

            }

            member

        }
        .flatMap { member ->

            if (member == null) {

                emptyList()
            }
            else {

                val memberClass = member.javaClass

                if (memberClass.isArray) {

                    (0 until Array.getLength(member))
                        .map { index ->

                            Array.get(member, index)
                        }
                        .filter {

                            couldBeResourceful(it.javaClass)

                        }

                }
                else {

                    if(couldBeResourceful(member.javaClass)) {

                        listOf(member)

                    }
                    else {

                        emptyList()

                    }

                }

            }

        }

private fun couldBeResourceful(memberClass: Class<*>): Boolean {

    val name = memberClass.name

    val excluded = memberClass.isPrimitive || arrayOf("java", "kotlin", "org.jblas").any {

        name.startsWith(it)

    }

    return !excluded

}