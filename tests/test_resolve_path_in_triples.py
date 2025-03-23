import unittest
from typing import List
from rdflib import BNode, Literal, RDF
from rdflib_r2r.sql_converter import resolve_paths_in_triples
from rdflib_r2r.types import SearchQuery
from tests.test_sql_converter import DEMO_NS
from rdflib.paths import SequencePath, AlternativePath, InvPath

def check(triples:List[SearchQuery], resolved_triples:List[List[SearchQuery]]):
    actual_triples_lists = list(resolve_paths_in_triples(triples, lambda x: False))
    for i, (ac, exp) in enumerate(zip(actual_triples_lists, resolved_triples)):
        bnode_map = {}
        def error():
            raise Exception(f"Error in triples list {i}:\nActual: {ac}\nExpected: {exp}")
        for at, bt in zip(ac, exp):
            for an, bn in zip(at,bt):
                if an in bnode_map:
                    if bnode_map[an] != bn:
                        error()
                elif isinstance(an, BNode):
                    if isinstance(bn, BNode) and bn not in bnode_map:
                        bnode_map[an] = bn
                        bnode_map[bn] = an
                else:
                    if an != bn:
                        error()
                        

def test_simple():
    check([(DEMO_NS.Me, RDF.type, DEMO_NS.Person)],[[(DEMO_NS.Me, RDF.type, DEMO_NS.Person)]])

def test_simple_multiple_triples():
    check([(DEMO_NS.Me, RDF.type, DEMO_NS.Person),(DEMO_NS.Me, DEMO_NS.name, Literal("MyName"))],
                [[(DEMO_NS.Me, RDF.type, DEMO_NS.Person),(DEMO_NS.Me, DEMO_NS.name, Literal("MyName"))]])

def test_sequence():
    b = BNode()
    check([(DEMO_NS.Me, SequencePath(DEMO_NS.dog, DEMO_NS.name), Literal("DogsName"))],
                [[(DEMO_NS.Me, DEMO_NS.dog, b),(b, DEMO_NS.name, Literal("DogsName"))]])

def test_alt():
    check([(DEMO_NS.Me, AlternativePath(DEMO_NS.dog, DEMO_NS.cat), DEMO_NS.MyPet)],
                [[(DEMO_NS.Me, DEMO_NS.dog, DEMO_NS.MyPet)], [(DEMO_NS.Me, DEMO_NS.cat, DEMO_NS.MyPet)]])

def test_inv():
    check([(DEMO_NS.MyDog, InvPath(DEMO_NS.dog), DEMO_NS.Me)],
                [[(DEMO_NS.Me, DEMO_NS.dog, DEMO_NS.MyDog)]])
    
def test_combo():
    b = BNode()
    check([(DEMO_NS.Me, SequencePath(AlternativePath(DEMO_NS.dog, DEMO_NS.cat), DEMO_NS.name), Literal("PetsName")),
                (DEMO_NS.Me, InvPath(DEMO_NS.master), DEMO_NS.MyDog)],
                    [[
                        (DEMO_NS.Me, DEMO_NS.dog, b),
                        (b, DEMO_NS.name, Literal("PetsName")),
                        (DEMO_NS.MyDog, DEMO_NS.master, DEMO_NS.Me)
                        ],
                    [
                        (DEMO_NS.Me, DEMO_NS.cat, b),
                        (b, DEMO_NS.name, Literal("PetsName")),
                        (DEMO_NS.MyDog, DEMO_NS.master, DEMO_NS.Me)
                        ]
                ])
