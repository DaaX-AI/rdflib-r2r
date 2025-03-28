"""
All commons types found in the rdflib_r2r package
"""
from re import S
from typing import Optional, Sequence, Set, Tuple, Union, Any, NamedTuple
from rdflib import Literal, URIRef, Variable, BNode
from rdflib.paths import Path
from sqlalchemy import CompoundSelect, Select

SPARQLVariable = Variable | BNode
AnyTerm = Union[URIRef, Literal,SPARQLVariable]
Triple = Tuple[URIRef, URIRef, AnyTerm]
TriplePattern = Union[URIRef, Literal, SPARQLVariable]
SearchQuery = Tuple[URIRef|SPARQLVariable, URIRef|SPARQLVariable|Path, AnyTerm]
BGP = Sequence[SearchQuery]
SQLQuery = Select|CompoundSelect