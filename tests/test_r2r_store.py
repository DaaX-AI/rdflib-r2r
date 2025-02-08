import logging
import re
from typing import Mapping
import unittest
from sqlalchemy import create_engine, Engine, Connection, text

from rdflib import Graph, URIRef
from rdflib_r2r.new_r2r_store import NewR2rStore
from rdflib_r2r.r2r_mapping import R2RMapping

def norm_ws(s:str|None) -> str|None:
    if s is None:
        return None
    return re.sub(r'\s+', ' ', s).strip()

class TestR2RStore(unittest.TestCase):

    db: Engine
    conn: Connection
    mapping: R2RMapping
    ns_map: Mapping[str, URIRef]

    @classmethod
    def setUpClass(cls):
        cls.db = create_engine("sqlite+pysqlite:///:memory:")
        cls.conn = cls.db.connect()
        with open('tests/data/CreateTables.sql') as f:
            for stmt in f.read().split(';'):
                cls.conn.execute(text(stmt))
        r2rg = Graph()
        r2rg.parse('tests/data/GoInspireR2RML.ttl')
        cls.mapping = R2RMapping(r2rg)
        cls.ns_map = {}
        for prefix, ns in r2rg.namespaces():
            cls.ns_map[prefix] = URIRef(ns)

    def setUp(self):
        self.store = NewR2rStore(self.db, self.mapping)

    def check(self, sparql:str, expected_sql:str|None):
        print("SPARQL:", sparql)
        print("Expected SQL:", expected_sql)
        actual_sql = self.store.getSQL(sparql, initNs=self.ns_map)
        print("Actual SQL:", actual_sql)
        self.assertEqual(norm_ws(actual_sql), norm_ws(expected_sql))

    def test_concrete_order_value(self):
        self.check('select ?v { <http://daax.ai/GoInspire/sample/order_header/Order_ID/1#this> gi:orderValue ?v}',
                   'SELECT t0."Order_Value" AS v\nFROM order_header AS t0\nWHERE t0."Order_ID" = 1')

    def test_concrete_order_concrete_value(self):
        self.check('select (1 as ?k) { <http://daax.ai/GoInspire/sample/order_header/Order_ID/1#this> gi:orderValue 3.50}',
                   'SELECT 1 AS k\nFROM order_header AS t0\nWHERE t0."Order_ID" = 1 AND t0."Order_Value" = 3.50')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()