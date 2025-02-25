import logging
import re
from typing import List, Mapping, Type
import unittest
from sqlalchemy import create_engine, Engine, Connection, text

from rdflib import RDF, BNode, Graph, Literal, Namespace, URIRef
from rdflib_r2r.new_r2r_store import NewR2rStore, resolve_paths_in_triples
from rdflib_r2r.r2r_store import SQL_FUNC
from rdflib_r2r.types import SearchQuery
from rdflib.paths import SequencePath, AlternativePath, InvPath, MulPath

def norm_ws(s:str|None) -> str|None:
    if s is None:
        return None
    return re.sub(r'\s+', ' ', s).strip()

DEMO_NS = Namespace("http://localhost:8890/Demo/")
OD_NS = Namespace(DEMO_NS["orders/"])

class TestR2RStore(unittest.TestCase):

    db: Engine
    conn: Connection
    mapping_graph: Graph
    ns_map: Mapping[str, URIRef]

    @staticmethod
    def setup_db(target:"TestR2RStore|Type[TestR2RStore]"):
        target.db = create_engine("sqlite+pysqlite:///:memory:")
        target.conn = target.db.connect()
        with open('tests/northwind/Northwind.sql') as f:
            for stmt in f.read().split(';'):
                target.conn.execute(text(stmt))
        target.mapping_graph = Graph().parse('tests/northwind/NorthwindR2RML.ttl')
        target.ns_map = {}
        for prefix, ns in target.mapping_graph.namespaces():
            target.ns_map[prefix] = URIRef(ns)
        
        target.ns_map['sqlf'] = SQL_FUNC

    def setUp(self):
        TestR2RStore.setup_db(self)
        self.store = NewR2rStore(self.db, self.mapping_graph)
        self.maxDiff = None

    def check(self, sparql:str, expected_sql:str|None):
        # print("SPARQL:", sparql)
        # print("Expected SQL:", expected_sql)
        actual_sql = self.store.getSQL(sparql, initNs=self.ns_map)
        # print("Actual SQL:", actual_sql)
        self.assertEqual(norm_ws(expected_sql), norm_ws(actual_sql))

    def test_order_value_by_id(self):
        self.check(f'select ?v {{ ?o a Demo:Orders; Demo:orderid 1; Demo:freight ?v}}',
                   'SELECT t0."Freight" AS v FROM "Orders" AS t0\nWHERE t0."OrderID" = 1')

    def test_concrete_order_value(self):
        self.check(f'select ?v {{ <http://localhost:8890/Demo/orders/1> Demo:freight ?v}}',
                   'SELECT t0."Freight" AS v FROM "Orders" AS t0\nWHERE t0."OrderID" = 1')
        
    def test_concrete_order_concrete_value(self):
        self.check(f'select (1 as ?k) {{ <http://localhost:8890/Demo/orders/1> Demo:freight 3.50}}',
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
            '''SELECT concat('http://localhost:8890/Demo/employees/', t0."EmployeeID") AS person FROM "Employees" AS t0
            UNION ALL SELECT concat('http://localhost:8890/Demo/customers/', t0."CustomerID") AS person FROM "Customers" AS t0'''
            )

    def test_union3(self):
        self.check('''select ?person_or_supplier { 
                   { ?person_or_supplier a Demo:Employees } UNION { ?person_or_supplier a Demo:Customers } UNION { ?person_or_supplier a Demo:Suppliers }
                   }''',
           '''SELECT concat('http://localhost:8890/Demo/employees/', t0."EmployeeID") AS person_or_supplier FROM "Employees" AS t0
           UNION ALL SELECT concat('http://localhost:8890/Demo/customers/', t0."CustomerID") AS person_or_supplier FROM "Customers" AS t0
           UNION ALL SELECT concat('http://localhost:8890/Demo/suppliers/', t0."SupplierID") AS person_or_supplier FROM "Suppliers" AS t0'''
            )

    def test_shared_prop(self):
        self.check('''select ?o { ?o Demo:city "Atlanta"}''',
                   '''SELECT concat('http://localhost:8890/Demo/customers/', t0."CustomerID") AS o FROM "Customers" AS t0 WHERE t0."City" = 'Atlanta'
                   UNION ALL SELECT concat('http://localhost:8890/Demo/employees/', t0."EmployeeID") AS o FROM "Employees" AS t0 WHERE t0."City" = 'Atlanta'
                   UNION ALL SELECT concat('http://localhost:8890/Demo/suppliers/', t0."SupplierID") AS o FROM "Suppliers" AS t0 WHERE t0."City" = 'Atlanta' ''')
        
    def test_shared_prop_with_class(self):
        self.check('''select ?o { ?o a Demo:Customers; Demo:city "Atlanta"}''',
                   '''SELECT concat('http://localhost:8890/Demo/customers/', t0."CustomerID") AS o FROM "Customers" AS t0 WHERE t0."City" = 'Atlanta' ''')
        
    def test_sparql_join(self):
        self.check('''select ?cn ?cc { { ?c a Demo:Customers; Demo:companyname ?cn . } { ?c Demo:city ?cc } }''', 
                    '''SELECT j1.cn AS cn, j2.cc AS cc FROM 
                        (SELECT concat('http://localhost:8890/Demo/customers/', t0."CustomerID") AS c, 
                            t0."CompanyName" AS cn FROM "Customers" AS t0) 
                        AS j1, 
                        (SELECT concat('http://localhost:8890/Demo/customers/', t0."CustomerID") AS c, 
                                t0."City" AS cc FROM "Customers" AS t0 
                            UNION ALL SELECT concat('http://localhost:8890/Demo/employees/', t0."EmployeeID") AS c, 
                                t0."City" AS cc FROM "Employees" AS t0 
                            UNION ALL SELECT concat('http://localhost:8890/Demo/suppliers/', t0."SupplierID") AS c, 
                                t0."City" AS cc FROM "Suppliers" AS t0) 
                        AS j2 
                        WHERE j1.c = j2.c'''
                    #'''SELECT j1.cn AS cn, j2.cc AS cc FROM 
                    #   (SELECT t0."CustomerID" AS c_CustomerID, t0."CompanyName" AS cn FROM "Customers" AS t0) 
                    #   AS j1,
                    #   (SELECT t0."CustomerID" AS c_CustomerID, t0."City" AS cc FROM "Customers" AS t0) 
                    #   AS j2,
                    #   WHERE j1.c = j2.c'''
                )

    def test_orderby_limit(self):
        self.check('''select ?order_date {
                     ?o a Demo:Orders; Demo:orderdate ?order_date; Demo:freight ?fr.
                     } order by ?fr limit 5''',
                     '''SELECT t0."OrderDate" AS order_date 
                     FROM "Orders" AS t0 
                     ORDER BY t0."Freight" LIMIT 5 OFFSET 0''')

    def test_orderby_desc_limit_offset(self):
        self.check('''select ?order_date {
                     ?o a Demo:Orders; Demo:orderdate ?order_date; Demo:freight ?fr; Demo:shippeddate ?sd.
                     } order by ?fr desc(?sd) limit 5 offset 10''',
                     '''SELECT t0."OrderDate" AS order_date 
                     FROM "Orders" AS t0 
                     ORDER BY t0."Freight", t0."ShippedDate" DESC LIMIT 5 OFFSET 10''')
        
    def test_blank_node(self):
        self.check('''select ?shid ?fr { ?sh Demo:shipperid ?shid; Demo:shippers_of_orders [ Demo:freight ?fr ]. }''',
                   '''SELECT t0."ShipperID" AS shid, t1."Freight" AS fr 
                   FROM "Shippers" AS t0, "Orders" AS t1 
                   WHERE t0."ShipperID" = t1."ShipVia"''')
        
    def test_path(self):
        self.check('''select ?shid ?fr { ?sh Demo:shipperid ?shid; Demo:shippers_of_orders / Demo:freight ?fr . }''',
                   '''SELECT t0."ShipperID" AS shid, t1."Freight" AS fr 
                   FROM "Shippers" AS t0, "Orders" AS t1 
                   WHERE t0."ShipperID" = t1."ShipVia"''')
        
    def test_const_query(self):
        self.check('''select (1 as ?one) {}''', 'SELECT 1 AS one')

    def test_in_op(self):
        self.check('''select (1 in (1,2,3) as ?itsin) {}''', 'SELECT 1 IN (1, 2, 3) AS itsin')

    def test_aggregate_join(self):
        self.check('''select ?shid (sum(?fr) as ?total_fr) { ?sh Demo:shipperid ?shid; Demo:shippers_of_orders / Demo:freight ?fr. } group by ?shid''',
                   '''SELECT t0."ShipperID" AS shid, sum(t1."Freight") AS total_fr 
                   FROM "Shippers" AS t0, "Orders" AS t1 
                   WHERE t0."ShipperID" = t1."ShipVia" GROUP BY t0."ShipperID"''')

    def test_aggregate_join_count(self):
        self.check('''select ?shid (count(distinct ?city) as ?city_count) { ?sh Demo:shipperid ?shid; Demo:shippers_of_orders / Demo:shipcity ?city. } group by ?shid''',
                   '''SELECT t0."ShipperID" AS shid, count(DISTINCT t1."ShipCity") AS city_count 
                   FROM "Shippers" AS t0, "Orders" AS t1 
                   WHERE t0."ShipperID" = t1."ShipVia" GROUP BY t0."ShipperID"''')

    def test_aggregate_join_count_star(self):
        self.check('''select ?shid (count(*) as ?combo_count) { ?sh Demo:shipperid ?shid; Demo:shippers_of_orders / Demo:shipcity ?city. } group by ?shid''',
                   '''SELECT t0."ShipperID" AS shid, count(*) AS combo_count 
                   FROM "Shippers" AS t0, "Orders" AS t1 
                   WHERE t0."ShipperID" = t1."ShipVia" GROUP BY t0."ShipperID"''')
        
    def test_regex_to_like(self):
        self.check('''select ?city { ?o Demo:shipcity ?city. filter regex(?city, "^.A[b-c][^d-f%*].*$", "is") }''',
                   '''SELECT t0."ShipCity" AS city 
                   FROM "Orders" AS t0 
                   WHERE t0."ShipCity" LIKE '_A[b-c][^d-f%*]%' ''')
        
    def test_sql_func(self):
        self.check('''select ?city { ?o Demo:shipcity ?city. filter (sqlf:LOWER(?city) = "atlanta") }''',
                   '''SELECT t0."ShipCity" AS city 
                   FROM "Orders" AS t0 
                   WHERE LOWER(t0."ShipCity") = 'atlanta' ''')

    def test_case(self):
        self.check('''SELECT (IF(4 > 3, "Yes", IF(3 < 4, "Whut", "No")) AS ?r) { }''',
                   '''SELECT CASE WHEN (4 > 3) THEN 'Yes' WHEN (3 < 4) THEN 'Whut' ELSE 'No' END AS r''')
        
    # For issue #2
    def test_disappearing_select(self):
        self.check(
        '''
        SELECT (MAX(?Total_Freight) AS ?Highest_Freight_Amount)
        {
            {
                SELECT ?oh_OrderDate ?oh_ShipCity (SUM(?oh_Freight) AS ?Total_Freight)
                {
                    FILTER (?oh_OrderDate >= "2023-08-01" && ?oh_OrderDate <= "2024-07-31")
                    ?oh a Demo:Orders.
                    ?oh Demo:orderdate ?oh_OrderDate.
                    ?oh Demo:shipcity ?oh_ShipCity.
                    ?oh Demo:freight ?oh_Freight.
                }
                GROUP BY ?oh_OrderDate ?oh_ShipCity
            }
        }
        ''',
        '''
        SELECT max(anon_1."Total_Freight") AS "Highest_Freight_Amount" 
        FROM (SELECT t0."OrderDate" AS "oh_OrderDate", t0."ShipCity" AS "oh_ShipCity", sum(t0."Freight") AS "Total_Freight" 
            FROM "Orders" AS t0 
            WHERE (t0."OrderDate" >= '2023-08-01') AND (t0."OrderDate" <= '2024-07-31') 
            GROUP BY t0."OrderDate", t0."ShipCity") AS anon_1
    ''')
        
    @unittest.expectedFailure
    def test_exists(self):
        self.check(
            '''select ?sid { ?s a Demo:Shippers. ?s Demo:shipperid ?sid. filter exists { ?s Demo:shippers_of_orders ?o. ?o a Demo:Orders. } }''',
            '''SELECT t0."ShipperID" AS sid
            FROM "Shippers" AS t0 
            WHERE EXISTS (SELECT * FROM "Orders" AS t1 WHERE t0."ShipperID" = t1."ShipVia")'''
        )

    def test_join_two_selects(self):
        self.check('''select ?cn ?cc { 
                   {
                    select ?c ?cn  { 
                        ?c a Demo:Customers; Demo:companyname ?cn.  
                    }
                   }
                   {
                    select ?c ?cc  { 
                        ?c Demo:city ?cc 
                   } 
                  }
                }''', 
                    '''SELECT j1.cn AS cn, j2.cc AS cc FROM 
                        (SELECT concat('http://localhost:8890/Demo/customers/', t0."CustomerID") AS c, 
                            t0."CompanyName" AS cn FROM "Customers" AS t0) 
                        AS j1, 
                        (SELECT concat('http://localhost:8890/Demo/customers/', t0."CustomerID") AS c, 
                                t0."City" AS cc FROM "Customers" AS t0 
                            UNION ALL SELECT concat('http://localhost:8890/Demo/employees/', t0."EmployeeID") AS c, 
                                t0."City" AS cc FROM "Employees" AS t0 
                            UNION ALL SELECT concat('http://localhost:8890/Demo/suppliers/', t0."SupplierID") AS c, 
                                t0."City" AS cc FROM "Suppliers" AS t0) 
                        AS j2 
                        WHERE j1.c = j2.c'''
                )
        
    def test_join_select_with_aggregation(self):
        self.check('''
                select ?cn ?total_fr { 
                    ?s a Demo:Shippers; Demo:companyname ?cn.  
                   
                   {
                    select ?s (SUM(?fr) as ?total_fr)  { 
                        ?s Demo:shippers_of_orders / Demo:freight ?fr.
                   }
                  }
                }''', 
                #XXX We should be able to get rid of the anon1 select layer...
                '''
                SELECT j1.cn AS cn, j2.total_fr AS total_fr 
                FROM 
                (SELECT concat('http://localhost:8890/Demo/shippers/', t0."ShipperID") AS s, t0."CompanyName" AS cn 
                    FROM "Shippers" AS t0) AS j1, 
                (SELECT anon_1.s AS s, anon_1.total_fr AS total_fr FROM
                    (SELECT concat('http://localhost:8890/Demo/shippers/', t0."ShipperID") AS s, sum(t1."Freight") AS total_fr 
                    FROM "Shippers" AS t0, "Orders" AS t1 WHERE t0."ShipperID" = t1."ShipVia") AS anon_1) AS j2 
                    WHERE j1.s = j2.s
                ''')
        
    def test_join_select_with_aggregation_in_subquery(self):
        self.check('''
                select ?cn ?total_fr 
                { 
                    {
                        select ?cn ?total_fr 
                        {
                            ?s a Demo:Shippers; Demo:companyname ?cn.  
                        
                            {
                                select ?s (SUM(?fr) as ?total_fr)  
                                { 
                                    ?s Demo:shippers_of_orders / Demo:freight ?fr.
                                }
                            }
                        }
                    }
                }''', 
                #XXX We should be able to get rid of the anon1 select layer...
                '''
                SELECT j1.cn AS cn, j2.total_fr AS total_fr 
                FROM 
                (SELECT concat('http://localhost:8890/Demo/shippers/', t0."ShipperID") AS s, t0."CompanyName" AS cn 
                    FROM "Shippers" AS t0) AS j1, 
                (SELECT anon_1.s AS s, anon_1.total_fr AS total_fr FROM
                    (SELECT concat('http://localhost:8890/Demo/shippers/', t0."ShipperID") AS s, sum(t1."Freight") AS total_fr 
                        FROM "Shippers" AS t0, "Orders" AS t1 WHERE t0."ShipperID" = t1."ShipVia") AS anon_1) AS j2 
                    WHERE j1.s = j2.s
                ''')
        
    def test_if(self):
        self.check('''SELECT (IF(4 > 3, "Yes", IF(3 < 4, "Whut", "No")) AS ?r) { }''',
                   '''SELECT CASE WHEN (4 > 3) THEN 'Yes' WHEN (3 < 4) THEN 'Whut' ELSE 'No' END AS r''')
    def test_arithmetic(self):
        self.check(
            '''select (?t0_Freight * 1000 as ?grams) { ?t0 a Demo:Orders. ?t0 Demo:freight ?t0_Freight. }''',
            '''SELECT t0."Freight" * 1000 AS grams FROM "Orders" AS t0''')
    
    # Bug #3
    def test_multiple_multiplicative_ops(self):
        self.check(
            '''SELECT (2 * 3 / 5 AS ?Answer) {}''',
            '''SELECT CAST((2 * 3) / 5 AS REAL) AS "Answer"''')
        
    def test_multiple_additive_ops(self):
        self.check(
            '''SELECT (2 - 3 + 5 AS ?Answer) {}''',
            '''SELECT (2 - 3) + 5 AS "Answer"''')
        
    def test_having(self):
        self.check('''select ?shid (sum(?fr) as ?total_fr) { 
                   ?sh Demo:shipperid ?shid; Demo:shippers_of_orders ?o.
                   ?o Demo:freight ?fr; Demo:orderid ?oid. 
                   } group by ?shid  HAVING (COUNT(DISTINCT ?oid) > 1)''',
            '''SELECT t0."ShipperID" AS shid, sum(t1."Freight") AS total_fr 
                FROM "Shippers" AS t0, "Orders" AS t1 
                WHERE t0."ShipperID" = t1."ShipVia" GROUP BY t0."ShipperID" HAVING count(DISTINCT t1."OrderID") > 1''')
        
    def test_count_star(self):
        self.check('''SELECT (COUNT(*) AS ?ShippersWithMoreThanFiveOrders)
                {
                    FILTER (?TotalOrders > 5)
                    {
                        SELECT ?sh_ShipperID (COUNT(?oh_OrderID) AS ?TotalOrders)
                        {
                        ?sh a Demo:Shippers;
                            Demo:shipperid ?sh_ShipperID;
                            Demo:shippers_of_orders / Demo:orderid ?oh_OrderID.
                        }
                        GROUP BY ?sh_ShipperID
                    }
                }
                ''', 
                '''SELECT count(*) AS "ShippersWithMoreThanFiveOrders" 
                FROM 
                    (SELECT t0."ShipperID" AS "sh_ShipperID", count(t1."OrderID") AS "TotalOrders" 
                    FROM "Shippers" AS t0, "Orders" AS t1 
                    WHERE t0."ShipperID" = t1."ShipVia" 
                    GROUP BY t0."ShipperID") AS anon_1 
                WHERE anon_1."TotalOrders" > 5
                ''')
        
class TestResolvePathsInTriples(unittest.TestCase):
    def check(self, triples:List[SearchQuery], resolved_triples:List[List[SearchQuery]]):
        actual_triples_lists = list(resolve_paths_in_triples(triples))
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
                            

    def test_simple(self):
        self.check([(DEMO_NS.Me, RDF.type, DEMO_NS.Person)],[[(DEMO_NS.Me, RDF.type, DEMO_NS.Person)]])

    def test_simple_multiple_triples(self):
        self.check([(DEMO_NS.Me, RDF.type, DEMO_NS.Person),(DEMO_NS.Me, DEMO_NS.name, Literal("MyName"))],
                   [[(DEMO_NS.Me, RDF.type, DEMO_NS.Person),(DEMO_NS.Me, DEMO_NS.name, Literal("MyName"))]])

    def test_sequence(self):
        b = BNode()
        self.check([(DEMO_NS.Me, SequencePath(DEMO_NS.dog, DEMO_NS.name), Literal("DogsName"))],
                   [[(DEMO_NS.Me, DEMO_NS.dog, b),(b, DEMO_NS.name, Literal("DogsName"))]])

    def test_alt(self):
        self.check([(DEMO_NS.Me, AlternativePath(DEMO_NS.dog, DEMO_NS.cat), DEMO_NS.MyPet)],
                   [[(DEMO_NS.Me, DEMO_NS.dog, DEMO_NS.MyPet)], [(DEMO_NS.Me, DEMO_NS.cat, DEMO_NS.MyPet)]])

    def test_inv(self):
        self.check([(DEMO_NS.MyDog, InvPath(DEMO_NS.dog), DEMO_NS.Me)],
                   [[(DEMO_NS.Me, DEMO_NS.dog, DEMO_NS.MyDog)]])
        
    def test_combo(self):
        b = BNode()
        self.check([(DEMO_NS.Me, SequencePath(AlternativePath(DEMO_NS.dog, DEMO_NS.cat), DEMO_NS.name), Literal("PetsName")),
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

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()