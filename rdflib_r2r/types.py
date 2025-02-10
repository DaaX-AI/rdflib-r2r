"""
All commons types found in the rdflib_r2r package
"""
from re import S
from typing import Optional, Set, Tuple, Union, Any, NamedTuple
from rdflib import Literal, URIRef, Variable, BNode
from sqlalchemy import CompoundSelect, Select

SPARQLVariable = Variable | BNode
AnyTerm = Union[URIRef, Literal,SPARQLVariable]
Triple = Tuple[URIRef, URIRef, AnyTerm]
TriplePattern = Union[URIRef, Literal, SPARQLVariable]
SearchQuery = Tuple[Optional[URIRef|SPARQLVariable], Optional[URIRef|SPARQLVariable], Optional[AnyTerm]]
BGP = Set[SearchQuery]
SQLQuery = Select|CompoundSelect