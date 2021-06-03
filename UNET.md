

downsampling factors: 2 2 2 ⇒ product = 8 (that's what we call the stride)

196  ->  192                                     112  -> 108 (has to be a multiple of "stride") ⇒ 104

	  |  /2                                   ^
          v                                       |

          96  ->  92                      60  ->  56

                   | /2                    ^
                   v                       |

                   46  ->  42      34  ->  30

                            | /2    ^
                            v       |

                            21  ->  17
