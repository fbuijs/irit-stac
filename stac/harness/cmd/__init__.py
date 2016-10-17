"""
irit-rst-dt subcommands
"""

# Author: Eric Kow
# License: CeCILL-B (French BSD3)

from . import (clean,
               count,
               evaluate,
               gather,
               localModels,
               model,
               parse,
               preview,
               serve,
               stop,
               structuredLearning)

SUBCOMMANDS =\
    [
        gather,
        localModels,
        preview,
        evaluate,
        count,
        clean,
        model,
        parse,
        serve,
        stop,
        structuredLearning
    ]
