import logging
import re
from typing import List, Mapping, Type
import unittest
from sqlalchemy import create_engine, Engine, Connection, text

from rdflib import RDF, BNode, Graph, Literal, Namespace, URIRef
from rdflib_r2r.sql_converter import SQLConverter, resolve_paths_in_triples
from rdflib_r2r.conversion_utils import SQL_FUNC
from rdflib_r2r.types import SearchQuery
from rdflib.paths import SequencePath, AlternativePath, InvPath, MulPath

def norm_ws(s:str|None) -> str|None:
    if s is None:
        return None
    return re.sub(r'\s+', ' ', s).strip()

DEMO_NS = Namespace("http://localhost:8890/schemas/Demo/")
OD_NS = Namespace(DEMO_NS["orders/"])
rr = Namespace("http://www.w3.org/ns/r2rml#")
        
class TestResolvePathsInTriples(unittest.TestCase):
    def check(self, triples:List[SearchQuery], resolved_triples:List[List[SearchQuery]]):
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

class TestSQLConverter(unittest.TestCase):

    db: Engine
    conn: Connection
    mapping_graph: Graph
    ns_map: Mapping[str, URIRef]

    @staticmethod
    def setup_db(target:"TestSQLConverter|Type[TestSQLConverter]"):
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
        TestSQLConverter.setup_db(self)
        if self._testMethodName == "test_column_for_direct_path":
            self.patch_graph_for_test_column_for_direct_path()
        self.store = SQLConverter(self.db, self.mapping_graph)
        self.maxDiff = None

    def check(self, sparql:str, expected_sql:str|None):
        # print("SPARQL:", sparql)
        # print("Expected SQL:", expected_sql)
        actual_sql = self.store.getSQL(sparql, initNs=self.ns_map)
        # print("Actual SQL:", actual_sql)
        self.assertEqual(norm_ws(expected_sql), norm_ws(actual_sql))

    def test_order_value_by_id(self):
        self.check(f'select ?v {{ ?o a Demo:Orders; Demo:orderid 1; Demo:freight ?v}}',
                   'SELECT o."Freight" AS v FROM "Orders" AS o\nWHERE o."OrderID" = 1')

    def test_concrete_order_value(self):
        self.check(f'select ?v {{ <http://localhost:8890/Demo/orders/1> Demo:freight ?v}}',
                   'SELECT o."Freight" AS v FROM "Orders" AS o\nWHERE o."OrderID" = 1')
        
    def test_concrete_order_concrete_value(self):
        self.check(f'select (1 as ?k) {{ <http://localhost:8890/Demo/orders/1> Demo:freight 3.50}}',
                   'SELECT 1 AS k\nFROM "Orders" AS o\nWHERE o."OrderID" = 1 AND o."Freight" = 3.50')

    def test_look_up_by_value_without_class(self):
        self.check(f'select ?o {{ ?o Demo:freight 3.50}}',
                   '''SELECT concat('http://localhost:8890/Demo/orders/', o."OrderID") AS o
                   FROM "Orders" AS o\nWHERE o."Freight" = 3.50''')
        
    def test_look_up_by_value_and_return_one_prop(self):
        self.check(f'select ?sco {{ ?o Demo:freight 3.50; Demo:shipcountry ?sco }}',
                   '''SELECT o."ShipCountry" AS sco
                   FROM "Orders" AS o\nWHERE o."Freight" = 3.50''')

    def test_look_up_by_value_and_return_props(self):
        self.check(f'select ?sco ?sci {{ ?o Demo:freight 3.50; Demo:shipcountry ?sco; Demo:shipcity ?sci }}',
                   '''SELECT o."ShipCountry" AS sco, o."ShipCity" AS sci
                   FROM "Orders" AS o\nWHERE o."Freight" = 3.50''')

    def test_look_up_by_value_with_class(self):
        self.check(f'select ?o {{ ?o a Demo:Orders; Demo:freight 3.50}}',
                   '''SELECT concat('http://localhost:8890/Demo/orders/', o."OrderID") AS o
                   FROM "Orders" AS o\nWHERE o."Freight" = 3.50''')
        
    def test_shipped_same_day(self):
        self.check('select ?o { ?o a Demo:Orders; Demo:shippeddate ?d; Demo:orderdate ?d. }',
                    '''SELECT concat('http://localhost:8890/Demo/orders/', o."OrderID") AS o
                    FROM "Orders" AS o WHERE o."OrderDate" = o."ShippedDate"''')

    def test_join(self):
        self.check('''select ?shid ?fr { ?sh Demo:shipperid ?shid; Demo:shippers_of_orders ?o. ?o Demo:freight ?fr. }''',
                   '''SELECT sh."ShipperID" AS shid, o."Freight" AS fr 
                   FROM "Shippers" AS sh, "Orders" AS o 
                   WHERE sh."ShipperID" = o."ShipVia"''')

    def test_join_two_iris(self):
        self.check('''select ?sh ?o { ?sh Demo:shippers_of_orders ?o }''',
                   '''SELECT concat('http://localhost:8890/Demo/shippers/', sh."ShipperID") AS sh, concat('http://localhost:8890/Demo/orders/', o."OrderID") AS o
                   FROM "Shippers" AS sh, "Orders" AS o 
                   WHERE sh."ShipperID" = o."ShipVia"''')

    def test_join_two_iris_second_const(self):
        self.check('''select ?sh  { ?sh Demo:shippers_of_orders <http://localhost:8890/Demo/orders/1> }''',
                   '''SELECT concat('http://localhost:8890/Demo/shippers/', sh."ShipperID") AS sh
                   FROM "Shippers" AS sh, "Orders" AS o 
                   WHERE sh."ShipperID" = o."ShipVia" AND o."OrderID" = 1''')

    def test_join_with_where(self):
        self.check('''select ?shid ?d ?fr { ?sh Demo:shipperid ?shid; Demo:shippers_of_orders ?o. ?o Demo:shippeddate ?d; Demo:freight ?fr. }''',
                   '''SELECT sh."ShipperID" AS shid, o."ShippedDate" AS d, o."Freight" AS fr 
                   FROM "Shippers" AS sh, "Orders" AS o 
                   WHERE sh."ShipperID" = o."ShipVia"''')
        
    def test_filter(self):
        self.check(f'select ?o {{ ?o a Demo:Orders; Demo:freight ?fr. filter(?fr < 3.50) }}',
                   '''SELECT concat('http://localhost:8890/Demo/orders/', o."OrderID") AS o
                   FROM "Orders" AS o\nWHERE o."Freight" < 3.50''')
        
    def test_union(self):
        self.check('''select ?person { 
                   { ?person a Demo:Employees } UNION { ?person a Demo:Customers }
                   }''',
            '''SELECT concat('http://localhost:8890/Demo/employees/', person."EmployeeID") AS person FROM "Employees" AS person
            UNION ALL SELECT concat('http://localhost:8890/Demo/customers/', person."CustomerID") AS person FROM "Customers" AS person'''
            )

    def test_union3(self):
        self.check('''select ?person_or_supplier { 
                   { ?person_or_supplier a Demo:Employees } UNION { ?person_or_supplier a Demo:Customers } UNION { ?person_or_supplier a Demo:Suppliers }
                   }''',
           '''SELECT concat('http://localhost:8890/Demo/employees/', person_or_supplier."EmployeeID") AS person_or_supplier FROM "Employees" AS person_or_supplier
           UNION ALL SELECT concat('http://localhost:8890/Demo/customers/', person_or_supplier."CustomerID") AS person_or_supplier FROM "Customers" AS person_or_supplier
           UNION ALL SELECT concat('http://localhost:8890/Demo/suppliers/', person_or_supplier."SupplierID") AS person_or_supplier FROM "Suppliers" AS person_or_supplier'''
            )

    def test_shared_prop(self):
        self.check('''select ?o { ?o Demo:city "Atlanta"}''',
                   '''SELECT concat('http://localhost:8890/Demo/customers/', o."CustomerID") AS o FROM "Customers" AS o WHERE o."City" = 'Atlanta'
                   UNION ALL SELECT concat('http://localhost:8890/Demo/employees/', o."EmployeeID") AS o FROM "Employees" AS o WHERE o."City" = 'Atlanta'
                   UNION ALL SELECT concat('http://localhost:8890/Demo/suppliers/', o."SupplierID") AS o FROM "Suppliers" AS o WHERE o."City" = 'Atlanta' ''')
        
    def test_shared_prop_with_class(self):
        self.check('''select ?o { ?o a Demo:Customers; Demo:city "Atlanta"}''',
                   '''SELECT concat('http://localhost:8890/Demo/customers/', o."CustomerID") AS o FROM "Customers" AS o WHERE o."City" = 'Atlanta' ''')
        
    def test_sparql_join(self):
        self.check('''select ?cn ?cc { { ?c a Demo:Customers; Demo:companyname ?cn . } { ?c Demo:city ?cc } }''', 
                   '''
                    SELECT c."CompanyName" AS cn, anon_1.cc AS cc FROM "Customers" AS c, 
                    (SELECT concat('http://localhost:8890/Demo/customers/', c."CustomerID") AS c, c."City" AS cc 
                    FROM "Customers" AS c 
                    UNION ALL 
                    SELECT concat('http://localhost:8890/Demo/employees/', c."EmployeeID") AS c, c."City" AS cc 
                    FROM "Employees" AS c 
                    UNION ALL 
                    SELECT concat('http://localhost:8890/Demo/suppliers/', c."SupplierID") AS c, c."City" AS cc 
                    FROM "Suppliers" AS c) AS anon_1 
                    WHERE concat('http://localhost:8890/Demo/customers/', c."CustomerID") = anon_1.c
                    ''')
        
    def test_sparql_join_two_tables(self):
        self.check('''select (COUNT(*) AS ?count) {
                   {
                    ?o a Demo:Orders; Demo:orderid ?oid.
                   }
                   {
                    ?ol a Demo:Order_Details; Demo:order_details_has_orders / Demo:orderid ?oid.
                   }
                }''',
                #XXX Should be able to get rid of o0 because its primary key matches o.
                '''SELECT count(*) AS COUNT 
                FROM "Orders" AS o, "Order Details" AS ol, "Orders" AS o0 
                WHERE o."OrderID" = o0."OrderID" AND ol."OrderID" = o0."OrderID"
                ''')

    def test_orderby_limit(self):
        self.check('''select ?order_date {
                     ?o a Demo:Orders; Demo:orderdate ?order_date; Demo:freight ?fr.
                     } order by ?fr limit 5''',
                     '''SELECT o."OrderDate" AS order_date 
                     FROM "Orders" AS o 
                     ORDER BY o."Freight" LIMIT 5 OFFSET 0''')

    def test_orderby_desc_limit_offset(self):
        self.check('''select ?order_date {
                     ?o a Demo:Orders; Demo:orderdate ?order_date; Demo:freight ?fr; Demo:shippeddate ?sd.
                     } order by ?fr desc(?sd) limit 5 offset 10''',
                     '''SELECT o."OrderDate" AS order_date 
                     FROM "Orders" AS o 
                     ORDER BY o."Freight", o."ShippedDate" DESC LIMIT 5 OFFSET 10''')
        
    def test_blank_node(self):
        self.check('''select ?shid ?fr { ?sh Demo:shipperid ?shid; Demo:shippers_of_orders [ Demo:freight ?fr ]. }''',
                   '''SELECT sh."ShipperID" AS shid, o."Freight" AS fr 
                   FROM "Shippers" AS sh, "Orders" AS o 
                   WHERE sh."ShipperID" = o."ShipVia"''')
        
    def test_path(self):
        self.check('''select ?shid ?fr { ?sh Demo:shipperid ?shid; Demo:shippers_of_orders / Demo:freight ?fr . }''',
                   '''SELECT sh."ShipperID" AS shid, o."Freight" AS fr 
                   FROM "Shippers" AS sh, "Orders" AS o 
                   WHERE sh."ShipperID" = o."ShipVia"''')
        
    def test_const_query(self):
        self.check('''select (1 as ?one) {}''', 'SELECT 1 AS one')

    def test_in_op(self):
        self.check('''select (1 in (1,2,3) as ?itsin) {}''', 'SELECT 1 IN (1, 2, 3) AS itsin')

    def test_aggregate_join(self):
        self.check('''select ?shid (sum(?fr) as ?total_fr) { ?sh Demo:shipperid ?shid; Demo:shippers_of_orders / Demo:freight ?fr. } group by ?shid''',
                   '''SELECT sh."ShipperID" AS shid, sum(o."Freight") AS total_fr 
                   FROM "Shippers" AS sh, "Orders" AS o 
                   WHERE sh."ShipperID" = o."ShipVia" GROUP BY sh."ShipperID"''')

    def test_aggregate_join_count(self):
        self.check('''select ?shid (count(distinct ?city) as ?city_count) { ?sh Demo:shipperid ?shid; Demo:shippers_of_orders / Demo:shipcity ?city. } group by ?shid''',
                   '''SELECT sh."ShipperID" AS shid, count(DISTINCT o."ShipCity") AS city_count 
                   FROM "Shippers" AS sh, "Orders" AS o 
                   WHERE sh."ShipperID" = o."ShipVia" GROUP BY sh."ShipperID"''')

    def test_aggregate_join_count_star(self):
        self.check('''select ?shid (count(*) as ?combo_count) { ?sh Demo:shipperid ?shid; Demo:shippers_of_orders / Demo:shipcity ?city. } group by ?shid''',
                   '''SELECT sh."ShipperID" AS shid, count(*) AS combo_count 
                   FROM "Shippers" AS sh, "Orders" AS o 
                   WHERE sh."ShipperID" = o."ShipVia" GROUP BY sh."ShipperID"''')
        
    def test_regex_to_like(self):
        self.check('''select ?city { ?o Demo:shipcity ?city. filter regex(?city, "^.A[b-c][^d-f%*].*$", "is") }''',
                   '''SELECT o."ShipCity" AS city 
                   FROM "Orders" AS o 
                   WHERE o."ShipCity" LIKE '_A[b-c][^d-f%*]%' ''')
        
    def test_sql_func(self):
        self.check('''select ?city { ?o Demo:shipcity ?city. filter (sqlf:LOWER(?city) = "atlanta") }''',
                   '''SELECT o."ShipCity" AS city 
                   FROM "Orders" AS o 
                   WHERE LOWER(o."ShipCity") = 'atlanta' ''')

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
        FROM (SELECT oh."OrderDate" AS "oh_OrderDate", oh."ShipCity" AS "oh_ShipCity", sum(oh."Freight") AS "Total_Freight" 
            FROM "Orders" AS oh 
            WHERE (oh."OrderDate" >= '2023-08-01') AND (oh."OrderDate" <= '2024-07-31') 
            GROUP BY oh."OrderDate", oh."ShipCity") AS anon_1
    ''')

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
                '''
                SELECT anon_1.cn AS cn, anon_2.cc AS cc 
                FROM 
                    (SELECT concat('http://localhost:8890/Demo/customers/', c."CustomerID") AS c, c."CompanyName" AS cn, c."CustomerID" AS "c_K0" 
                    FROM "Customers" AS c) AS anon_1, 
                    (SELECT concat('http://localhost:8890/Demo/customers/', c."CustomerID") AS c, c."City" AS cc 
                    FROM "Customers" AS c 
                    UNION ALL 
                    SELECT concat('http://localhost:8890/Demo/employees/', c."EmployeeID") AS c, c."City" AS cc 
                    FROM "Employees" AS c 
                    UNION ALL 
                    SELECT concat('http://localhost:8890/Demo/suppliers/', c."SupplierID") AS c, c."City" AS cc 
                    FROM "Suppliers" AS c) 
                    AS anon_2 WHERE anon_1.c = anon_2.c
                '''
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
                '''
                SELECT s."CompanyName" AS cn, anon_1.total_fr AS total_fr 
                FROM "Shippers" AS s, 
                    (SELECT concat('http://localhost:8890/Demo/shippers/', s."ShipperID") AS s, sum(o."Freight") AS total_fr, s."ShipperID" AS "s_K0" 
                    FROM "Shippers" AS s, "Orders" AS o WHERE s."ShipperID" = o."ShipVia") AS anon_1 
                    WHERE s."ShipperID" = anon_1."s_K0"
                '''
                )
        
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
                #XXX We should be able to get rid URI packing/unpacking...
                '''
                SELECT anon_1.cn AS cn, anon_1.total_fr AS total_fr 
                FROM 
                    (SELECT s."CompanyName" AS cn, anon_2.total_fr AS total_fr 
                    FROM "Shippers" AS s, 
                        (SELECT concat('http://localhost:8890/Demo/shippers/', s."ShipperID") AS s, sum(o."Freight") AS total_fr, s."ShipperID" AS "s_K0" 
                        FROM "Shippers" AS s, "Orders" AS o 
                        WHERE s."ShipperID" = o."ShipVia") AS anon_2 
                    WHERE s."ShipperID" = anon_2."s_K0") AS anon_1
                '''
            )
        
    def test_if(self):
        self.check('''SELECT (IF(4 > 3, "Yes", IF(3 < 4, "Whut", "No")) AS ?r) { }''',
                   '''SELECT CASE WHEN (4 > 3) THEN 'Yes' WHEN (3 < 4) THEN 'Whut' ELSE 'No' END AS r''')
    def test_arithmetic(self):
        self.check(
            '''select (?o_Freight * 1000 as ?grams) { ?o a Demo:Orders. ?o Demo:freight ?o_Freight. }''',
            '''SELECT o."Freight" * 1000 AS grams FROM "Orders" AS o''')
    
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
            '''SELECT sh."ShipperID" AS shid, sum(o."Freight") AS total_fr 
                FROM "Shippers" AS sh, "Orders" AS o 
                WHERE sh."ShipperID" = o."ShipVia" GROUP BY sh."ShipperID" HAVING count(DISTINCT o."OrderID") > 1''')
        
    def test_subquery_in_aggregate(self):
        self.check('''SELECT (MAX(?fr) AS ?MaxFreight) {
                {
                        SELECT ?fr
                        {
                            ?o a Demo:Orders; Demo:freight ?fr.
                        }
                    }
                }
                ''', 
                '''SELECT max(anon_1.fr) AS "MaxFreight" 
                FROM 
                    (SELECT o."Freight" AS fr FROM "Orders" AS o) AS anon_1
                ''')
        
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
                    (SELECT sh."ShipperID" AS "sh_ShipperID", count(o."OrderID") AS "TotalOrders" 
                    FROM "Shippers" AS sh, "Orders" AS o 
                    WHERE sh."ShipperID" = o."ShipVia" 
                    GROUP BY sh."ShipperID") AS anon_1 
                WHERE anon_1."TotalOrders" > 5
                ''')
        

    def test_unbound(self):
        self.check(
        '''SELECT ?oh_OrderID { FILTER (!BOUND(?oh_Freight)) ?oh a Demo:Orders. ?oh Demo:freight ?oh_Freight. ?oh Demo:orderid ?oh_OrderID. }''',
        '''SELECT oh."OrderID" AS "oh_OrderID" FROM "Orders" AS oh WHERE oh."Freight" IS NULL'''
        )

    def test_bound(self):
        self.check(
        '''SELECT ?oh_OrderID { FILTER (BOUND(?oh_Freight)) ?oh a Demo:Orders. ?oh Demo:freight ?oh_Freight. ?oh Demo:orderid ?oh_OrderID. }''',
        '''SELECT oh."OrderID" AS "oh_OrderID" FROM "Orders" AS oh WHERE not(oh."Freight" IS NULL)'''
        )

    def test_exists(self):
        self.check(
            '''select ?sid { ?s a Demo:Shippers. ?s Demo:shipperid ?sid. filter exists { ?s Demo:shippers_of_orders ?o. ?o a Demo:Orders. } }''',
            '''SELECT s."ShipperID" AS sid
            FROM "Shippers" AS s 
            WHERE EXISTS (SELECT * FROM "Orders" AS o WHERE s."ShipperID" = o."ShipVia")'''
        )

    def test_simple_left_join(self):
        self.check(
            '''SELECT ?shid ?fr { ?s a Demo:Shippers. ?s Demo:shipperid ?shid. OPTIONAL { ?o a Demo:Orders. ?s Demo:shippers_of_orders ?o. ?o Demo:freight ?fr. }  }''',
                   '''SELECT s."ShipperID" AS shid, o."Freight" AS fr 
                   FROM "Shippers" AS s LEFT OUTER JOIN "Orders" AS o ON s."ShipperID" = o."ShipVia"'''
        )

    def test_left_join(self):
        self.check('''SELECT ?shid ?fr { ?s a Demo:Shippers. OPTIONAL { ?o a Demo:Orders. ?s Demo:shippers_of_orders ?o. ?o Demo:freight ?fr. } ?s Demo:shipperid ?shid. }''',
                   '''SELECT s."ShipperID" AS shid, o."Freight" AS fr 
                   FROM "Shippers" AS s LEFT OUTER JOIN "Orders" AS o ON s."ShipperID" = o."ShipVia"''')        
        
    def test_column_for_inverse_path(self):
        self.check('''SELECT ?oid ?shid { ?o a Demo:Orders; Demo:orderid ?oid; ^Demo:shippers_of_orders / Demo:shipperid ?shid }''',
                   '''SELECT o."OrderID" AS oid, o."ShipVia" AS shid FROM "Orders" AS o''')


    def patch_graph_for_test_column_for_direct_path(self):
        #1. Find the right object map.
        om = list(self.mapping_graph.query( 
            '''select ?om {
                ?tm rr:logicalTable [  rr:tableName "Order Details" ];
                  rr:predicateObjectMap [ rr:predicateMap [ rr:constant Demo:order_details_has_orders ] ; rr:objectMap ?om ]
            }
            '''
        ))[0][0] # type: ignore

        #2. Remove the template mapping
        self.mapping_graph.remove((om, rr.template, None))

        #3. Add join-based mapping instead.
        tm = list(self.mapping_graph.subjects(SequencePath(rr.logicalTable, rr.tableName), Literal("Orders")))[0]
        self.mapping_graph.add((om, rr.parentTriplesMap, tm))
        jc = BNode()
        self.mapping_graph.add((om, rr.joinCondition, jc))
        self.mapping_graph.add((jc, rr.child, Literal("OrderID")))
        self.mapping_graph.add((jc, rr.parent, Literal("OrderID")))

    def test_column_for_direct_path(self):
        # Note: graph patched in setUp
        self.check('''SELECT ?od ?oid { ?od a Demo:Order_Details; Demo:order_details_has_orders / Demo:orderid ?oid }''',
                   '''SELECT concat('http://localhost:8890/Demo/order_details/', od."OrderID", '/', od."ProductID") AS od, od."OrderID" AS oid FROM "Order Details" AS od''')


    def test_cast(self):
        self.check(
            'SELECT (xsd:decimal(42) AS ?flt42) { }',
            '''SELECT CAST(42 AS NUMERIC) AS flt42'''
            )    
        
    def test_left_join_with_eq(self):
        self.check(
        '''
        SELECT ?Order_Count ?Order_Count_0
        {
            ?OH a Demo:Orders; Demo:orderid ?Order_Count.
            OPTIONAL
            {
                FILTER (?Order_Count = ?Order_Count_0)
                {
                    ?OH_0 a Demo:Orders; Demo:orderid ?Order_Count_0.
                }
            }
        }
        ''',
        '''
        SELECT "OH"."OrderID" AS "Order_Count", "OH_0"."OrderID" AS "Order_Count_0" 
        FROM "Orders" AS "OH" 
        LEFT OUTER JOIN "Orders" AS "OH_0" 
        ON "OH"."OrderID" = "OH_0"."OrderID"
        '''
        )

    #Bug #6
    def test_named_subquery_expression_in_exists(self):
        self.check(
    '''
    SELECT (COUNT(*) AS ?Churned_Customers)
    {
        FILTER (!EXISTS {
            {
                SELECT ?fr
                {
                    ?order_header_0 a Demo:Orders;
                        Demo:freight ?fr.
                }
            }
        })
        {
            ?order_header a Demo:Orders;
                Demo:freight ?fr.
        }
    }''',
    '''
    SELECT count(*) AS "Churned_Customers" 
    FROM "Orders" AS order_header 
    WHERE NOT (EXISTS 
        (SELECT * FROM 
            (SELECT order_header_0."Freight" AS fr FROM "Orders" AS order_header_0)
            AS anon_1 WHERE anon_1.fr = order_header."Freight"))
    ''')
        
    #Bug #6
    #@unittest.expectedFailure
    def test_two_subqueries_one_named_in_exists(self):
        self.check(
    '''
    SELECT (COUNT(*) AS ?Churned_Customers)
    {
        FILTER (!EXISTS {
            {
                SELECT ?fr
                {
                    ?order_header_0 a Demo:Orders;
                        Demo:freight ?fr.
                }
            }
        })
        {
            select ?fr {
                ?order_header a Demo:Orders;
                    Demo:freight ?fr.
            }
        }
    }''',
    '''
    SELECT count(*) AS "Churned_Customers" 
    FROM (SELECT order_header."Freight" AS fr FROM "Orders" AS order_header) AS anon_1
    WHERE NOT (EXISTS 
        (SELECT * FROM 
            (SELECT order_header_0."Freight" AS fr FROM "Orders" AS order_header_0)
            AS anon_2 WHERE anon_2.fr = anon_1.fr))
    ''')

    def test_select_distinct_in_subquery(self):
        self.check(
    '''
    SELECT (COUNT(?oh_Customer_ID) AS ?Number_of_Customers_with_Purchases_in_2022)
    {
        {
            SELECT DISTINCT ?oh_Customer_ID
            {
            FILTER (?oh_OrderDate >= "2022-01-01" && ?oh_OrderDate <= "2022-12-31")
            ?oh a Demo:Orders;
                Demo:orderdate ?oh_OrderDate;
                ^Demo:customers_of_orders/Demo:customerid ?oh_Customer_ID.
            }
        }
    }
    ''',
    '''
    SELECT count(anon_1."oh_Customer_ID") AS "Number_of_Customers_with_Purchases_in_2022"
    FROM (SELECT DISTINCT oh."CustomerID" AS "oh_Customer_ID"
        FROM "Orders" AS oh
        WHERE (oh."OrderDate" >= '2022-01-01') AND (oh."OrderDate" <= '2022-12-31')) AS anon_1
    ''')
        
    def test_neg(self):
        self.check('''SELECT (-?t0_Freight AS ?neg_freight) { ?t0 a Demo:Orders. ?t0 Demo:freight ?t0_Freight. }''',
                   '''SELECT -t0."Freight" AS neg_freight FROM "Orders" AS t0''')        