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
        with open('tests/northwind/Northwind.sql') as f:
            for stmt in f.read().split(';'):
                target.conn.execute(text(stmt))
        r2rg = Graph()
        r2rg.parse('tests/northwind/NorthwindR2RML.ttl')
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
        
    def test_look_up_by_value_and_return_one_prop(self):
        self.check(f'select ?sco {{ ?o Demo:freight 3.50; Demo:shipcountry ?sco }}',
                   '''SELECT t0."ShipCountry" AS sco
                   FROM "Orders" AS t0\nWHERE t0."Freight" = 3.50''')

    def test_look_up_by_value_and_return_props(self):
        self.check(f'select ?sco ?sci {{ ?o Demo:freight 3.50; Demo:shipcountry ?sco; Demo:shipcity ?sci }}',
                   '''SELECT t0."ShipCountry" AS sco, t0."ShipCity" AS sci
                   FROM "Orders" AS t0\nWHERE t0."Freight" = 3.50''')

    def test_look_up_by_value_with_class(self):
        self.check(f'select ?o {{ ?o a Demo:Orders; Demo:freight 3.50}}',
                   '''SELECT concat('http://localhost:8890/Demo/orders/', t0."OrderID") AS o
                   FROM "Orders" AS t0\nWHERE t0."Freight" = 3.50''')
        
    def test_shipped_same_day(self):
        self.check('select ?o { ?o a Demo:Orders; Demo:shippeddate ?d; Demo:orderdate ?d. }',
                    '''SELECT concat('http://localhost:8890/Demo/orders/', t0."OrderID") AS o
                    FROM "Orders" AS t0 WHERE t0."OrderDate" = t0."ShippedDate"''')

    def test_join(self):
        self.check('''select ?shid ?fr { ?sh Demo:shipperid ?shid; Demo:shippers_of_orders ?o. ?o Demo:freight ?fr. }''',
                   '''SELECT t0."ShipperID" AS shid, t1."Freight" AS fr 
                   FROM "Shippers" AS t0, "Orders" AS t1 
                   WHERE t0."ShipperID" = t1."ShipVia"''')

    def test_join_two_iris(self):
        self.check('''select ?sh ?o { ?sh Demo:shippers_of_orders ?o }''',
                   '''SELECT concat('http://localhost:8890/Demo/shippers/', t0."ShipperID") AS sh, concat('http://localhost:8890/Demo/orders/', t1."OrderID") AS o
                   FROM "Shippers" AS t0, "Orders" AS t1 
                   WHERE t0."ShipperID" = t1."ShipVia"''')

    def test_join_two_iris_second_const(self):
        self.check('''select ?sh  { ?sh Demo:shippers_of_orders <http://localhost:8890/Demo/orders/1> }''',
                   '''SELECT concat('http://localhost:8890/Demo/shippers/', t0."ShipperID") AS sh
                   FROM "Shippers" AS t0, "Orders" AS t1 
                   WHERE t0."ShipperID" = t1."ShipVia" AND t1."OrderID" = 1''')

    def test_join_with_where(self):
        self.check('''select ?shid ?d ?fr { ?sh Demo:shipperid ?shid; Demo:shippers_of_orders ?o. ?o Demo:shippeddate ?d; Demo:freight ?fr. }''',
                   '''SELECT t0."ShipperID" AS shid, t1."ShippedDate" AS d, t1."Freight" AS fr 
                   FROM "Shippers" AS t0, "Orders" AS t1 
                   WHERE t0."ShipperID" = t1."ShipVia"''')
        
    def test_filter(self):
        self.check(f'select ?o {{ ?o a Demo:Orders; Demo:freight ?fr. filter(?fr < 3.50) }}',
                   '''SELECT concat('http://localhost:8890/Demo/orders/', t0."OrderID") AS o
                   FROM "Orders" AS t0\nWHERE t0."Freight" < 3.50''')
        
    def test_union(self):
        self.check('''select ?person { 
                   { ?person a Demo:Employees } UNION { ?person a Demo:Customers }
                   }''',
            '''SELECT concat('http://localhost:8890/Demo/employees/', t0."EmployeeID") as person FROM "Employees" AS t0
            UNION SELECT concat('http://localhost:8890/Demo/suppliers/', t0."SupplierID") as o FROM "Suppliers" AS t0''')

    def test_shared_prop(self):
        self.check('''select ?o { ?o Demo:city "Atlanta"}''',
                   '''SELECT concat('http://localhost:8890/Demo/customers/', t0."CustomerID") as o FROM "Customers" AS t0 WHERE t0."City" = 'Atlanta'
                   UNION SELECT concat('http://localhost:8890/Demo/employees/', t0."EmployeeID") as o FROM "Employees" AS t0 WHERE t0."City" = 'Atlanta'
                   UNION SELECT concat('http://localhost:8890/Demo/suppliers/', t0."SupplierID") as o FROM "Suppliers" AS t0 WHERE t0."City" = 'Atlanta' ''')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()