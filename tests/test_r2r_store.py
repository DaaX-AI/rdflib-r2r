import logging
import re
from typing import Mapping, Type
import unittest
from sqlalchemy import create_engine, Engine, Connection, text

from rdflib import Graph, Namespace, URIRef
from rdflib_r2r.new_r2r_store import NewR2rStore
from rdflib_r2r.r2r_mapping import R2RMapping

def norm_ws(s:str|None) -> str|None:
    if s is None:
        return None
    return re.sub(r'\s+', ' ', s).strip()

DEMO_NS = Namespace("http://localhost:8890/Demo/")
OD_NS = Namespace(DEMO_NS["orders/"])

class TestR2RStore(unittest.TestCase):

    db: Engine
    conn: Connection
    mapping: R2RMapping
    ns_map: Mapping[str, URIRef]

    @staticmethod
    def setup_db(target:"TestR2RStore|Type[TestR2RStore]"):
        target.db = create_engine("sqlite+pysqlite:///:memory:")
        target.conn = target.db.connect()
        with open('tests/data/Northwind.sql') as f:
            for stmt in f.read().split(';'):
                target.conn.execute(text(stmt))
        r2rg = Graph()
        r2rg.parse('tests/data/NorthwindR2RML.ttl')
        target.mapping = R2RMapping(r2rg)
        target.ns_map = {}
        for prefix, ns in r2rg.namespaces():
            target.ns_map[prefix] = URIRef(ns)
        

    def setUp(self):
        TestR2RStore.setup_db(self)
        self.store = NewR2rStore(self.db, self.mapping)

    def check(self, sparql:str, expected_sql:str|None):
        # print("SPARQL:", sparql)
        # print("Expected SQL:", expected_sql)
        actual_sql = self.store.getSQL(sparql, initNs=self.ns_map)
        # print("Actual SQL:", actual_sql)
        self.assertEqual(norm_ws(actual_sql), norm_ws(expected_sql))

    def test_concrete_order_value(self):
        self.check(f'select ?v {{ <{OD_NS}1> Demo:freight ?v}}',
                   'SELECT t0."Freight" AS v FROM "Orders" AS t0\nWHERE t0."OrderID" = 1')

    def test_concrete_order_concrete_value(self):
        self.check(f'select (1 as ?k) {{ <{OD_NS}1> Demo:freight 3.50}}',
                   'SELECT 1 AS k\nFROM "Orders" AS t0\nWHERE t0."OrderID" = 1 AND t0."Freight" = 3.50')

    def test_look_up_by_value_without_class(self):
        self.check(f'select ?o {{ ?o Demo:freight 3.50}}',
                   '''SELECT concat('http://localhost:8890/Demo/orders/', t0."OrderID") AS o
                   FROM "Orders" AS t0\nWHERE t0."Freight" = 3.50''')

    def test_look_up_by_value_with_class(self):
        self.check(f'select ?o {{ ?o a Demo:Orders; Demo:freight 3.50}}',
                   '''SELECT concat('http://localhost:8890/Demo/orders/', t0."OrderID") AS o
                   FROM "Orders" AS t0\nWHERE t0."Freight" = 3.50''')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()