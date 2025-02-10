"""
All commons types found in the rdflib_r2r package
"""
from typing import Optional, Set, Tuple, Union, Any, NamedTuple
from rdflib import Literal, URIRef, Variable
from sqlalchemy import CompoundSelect, Select

AnyTerm = Union[URIRef, Literal,Variable]
Triple = Tuple[URIRef, URIRef, AnyTerm]
TriplePattern = Union[URIRef, Literal, Variable]
SearchQuery = Tuple[Optional[URIRef|Variable], Optional[URIRef|Variable], Optional[AnyTerm]]
BGP = Set[SearchQuery]
SQLQuery = Select|CompoundSelect