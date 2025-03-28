import logging
import re
from typing import Callable, Generator, List, Mapping, Type
import unittest
import pytest
from sqlalchemy import create_engine, Engine, Connection, text

from rdflib import RDF, RDFS, BNode, Graph, Literal, Namespace, URIRef
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
        
        
class BaseSQLConvertingTest(unittest.TestCase):
    db: Engine
    conn: Connection
    mapping_graph: Graph
    ns_map: Mapping[str, URIRef]

    @staticmethod
    def setup_db(target:"BaseSQLConvertingTest|Type[BaseSQLConvertingTest]"):
        target.db = create_engine("sqlite+pysqlite:///:memory:")
        target.conn = target.db.connect()
        with open('tests/northwind/Northwind.sql') as f:
            for stmt in f.read().split(';'):
                target.conn.execute(text(stmt))
        target.conn.commit()
        target.mapping_graph = Graph().parse('tests/northwind/NorthwindR2RML.ttl')
        target.ns_map = {}
        for prefix, ns in target.mapping_graph.namespaces():
            target.ns_map[prefix] = URIRef(ns)
        
        target.ns_map['sqlf'] = SQL_FUNC

    def setUp(self):
        BaseSQLConvertingTest.setup_db(self)
        self.maxDiff = None

@pytest.fixture
def mapping_graph() -> Graph:
    g = Graph()
    g.parse('tests/northwind/NorthwindR2RML.ttl')
    return g

@pytest.fixture
def test_db() -> Generator[Engine, None, None]:
    db = create_engine("sqlite+pysqlite:///:memory:")
    conn = db.connect()
    with open('tests/northwind/Northwind.sql') as f:
        for stmt in f.read().split(';'):
            conn.execute(text(stmt))
    conn.commit()
    yield db
    conn.close()

@pytest.fixture
def converter(test_db:Engine, mapping_graph:Graph) -> SQLConverter:
    return SQLConverter(test_db, mapping_graph)

class Checker:
    def __init__(self, converter:SQLConverter):
        self.converter = converter
        self.ns_map = { **{ p:ns for p,ns in self.converter.mapping_graph.namespaces() }, 'sqlf': SQL_FUNC }

    def __call__(self, sparql:str, expected_sql:str|None):
        actual_sql = self.converter.getSQL(sparql, initNs=self.ns_map)
        assert norm_ws(expected_sql) == norm_ws(actual_sql)

@pytest.fixture
def check(converter:SQLConverter) -> Callable[[str, str|None], None]:
    return Checker(converter)

def test_order_value_by_id(check:Callable[[str, str|None], None]):
    check(f'select ?v {{ ?o a Demo:Orders; Demo:orderid 1; Demo:freight ?v}}',
                'SELECT o."Freight" AS v FROM "Orders" AS o\nWHERE o."OrderID" = 1')

def test_concrete_order_value(check:Callable[[str, str|None], None]):
    check(f'select ?v {{ <http://localhost:8890/Demo/orders/1> Demo:freight ?v}}',
                'SELECT o."Freight" AS v FROM "Orders" AS o\nWHERE o."OrderID" = 1')
    
def test_concrete_order_concrete_value(check):
    check(f'select (1 as ?k) {{ <http://localhost:8890/Demo/orders/1> Demo:freight 3.50}}',
                'SELECT 1 AS k\nFROM "Orders" AS o\nWHERE o."OrderID" = 1 AND o."Freight" = 3.50')

def test_look_up_by_value_without_class(check):
    check(f'select ?o {{ ?o Demo:freight 3.50}}',
                '''SELECT concat('http://localhost:8890/Demo/orders/', o."OrderID") AS o
                FROM "Orders" AS o\nWHERE o."Freight" = 3.50''')
    
def test_look_up_by_value_and_return_one_prop(check):
    check(f'select ?sco {{ ?o Demo:freight 3.50; Demo:shipcountry ?sco }}',
                '''SELECT o."ShipCountry" AS sco
                FROM "Orders" AS o\nWHERE o."Freight" = 3.50''')

def test_look_up_by_value_and_return_props(check):
    check(f'select ?sco ?sci {{ ?o Demo:freight 3.50; Demo:shipcountry ?sco; Demo:shipcity ?sci }}',
                '''SELECT o."ShipCountry" AS sco, o."ShipCity" AS sci
                FROM "Orders" AS o\nWHERE o."Freight" = 3.50''')

def test_look_up_by_value_with_class(check):
    check(f'select ?o {{ ?o a Demo:Orders; Demo:freight 3.50}}',
                '''SELECT concat('http://localhost:8890/Demo/orders/', o."OrderID") AS o
                FROM "Orders" AS o\nWHERE o."Freight" = 3.50''')
    
def test_shipped_same_day(check):
    check('select ?o { ?o a Demo:Orders; Demo:shippeddate ?d; Demo:orderdate ?d. }',
                '''SELECT concat('http://localhost:8890/Demo/orders/', o."OrderID") AS o
                FROM "Orders" AS o WHERE o."OrderDate" = o."ShippedDate"''')

def test_join(check):
    check('''select ?shid ?fr { ?sh Demo:shipperid ?shid; Demo:shippers_of_orders ?o. ?o Demo:freight ?fr. }''',
                '''SELECT sh."ShipperID" AS shid, o."Freight" AS fr 
                FROM "Shippers" AS sh, "Orders" AS o 
                WHERE sh."ShipperID" = o."ShipVia"''')

def test_join_two_iris(check):
    check('''select ?sh ?o { ?sh Demo:shippers_of_orders ?o }''',
                '''SELECT concat('http://localhost:8890/Demo/shippers/', sh."ShipperID") AS sh, concat('http://localhost:8890/Demo/orders/', o."OrderID") AS o
                FROM "Shippers" AS sh, "Orders" AS o 
                WHERE sh."ShipperID" = o."ShipVia"''')

def test_join_two_iris_second_const(check):
    check('''select ?sh  { ?sh Demo:shippers_of_orders <http://localhost:8890/Demo/orders/1> }''',
                '''SELECT concat('http://localhost:8890/Demo/shippers/', sh."ShipperID") AS sh
                FROM "Shippers" AS sh, "Orders" AS o 
                WHERE sh."ShipperID" = o."ShipVia" AND o."OrderID" = 1''')

def test_join_with_where(check):
    check('''select ?shid ?d ?fr { ?sh Demo:shipperid ?shid; Demo:shippers_of_orders ?o. ?o Demo:shippeddate ?d; Demo:freight ?fr. }''',
                '''SELECT sh."ShipperID" AS shid, o."ShippedDate" AS d, o."Freight" AS fr 
                FROM "Shippers" AS sh, "Orders" AS o 
                WHERE sh."ShipperID" = o."ShipVia"''')
    
def test_filter(check):
    check(f'select ?o {{ ?o a Demo:Orders; Demo:freight ?fr. filter(?fr < 3.50) }}',
                '''SELECT concat('http://localhost:8890/Demo/orders/', o."OrderID") AS o
                FROM "Orders" AS o\nWHERE o."Freight" < 3.50''')
    
def test_union(check):
    check('''select ?person { 
                { ?person a Demo:Employees } UNION { ?person a Demo:Customers }
                }''',
        '''SELECT concat('http://localhost:8890/Demo/employees/', person."EmployeeID") AS person FROM "Employees" AS person
        UNION ALL SELECT concat('http://localhost:8890/Demo/customers/', person."CustomerID") AS person FROM "Customers" AS person'''
        )

def test_union3(check):
    check('''select ?person_or_supplier { 
                { ?person_or_supplier a Demo:Employees } UNION { ?person_or_supplier a Demo:Customers } UNION { ?person_or_supplier a Demo:Suppliers }
                }''',
        '''SELECT concat('http://localhost:8890/Demo/employees/', person_or_supplier."EmployeeID") AS person_or_supplier FROM "Employees" AS person_or_supplier
        UNION ALL SELECT concat('http://localhost:8890/Demo/customers/', person_or_supplier."CustomerID") AS person_or_supplier FROM "Customers" AS person_or_supplier
        UNION ALL SELECT concat('http://localhost:8890/Demo/suppliers/', person_or_supplier."SupplierID") AS person_or_supplier FROM "Suppliers" AS person_or_supplier'''
        )

def test_shared_prop(check):
    check('''select ?o { ?o Demo:city "Atlanta"}''',
                '''SELECT concat('http://localhost:8890/Demo/customers/', o."CustomerID") AS o FROM "Customers" AS o WHERE o."City" = 'Atlanta'
                UNION ALL SELECT concat('http://localhost:8890/Demo/employees/', o."EmployeeID") AS o FROM "Employees" AS o WHERE o."City" = 'Atlanta'
                UNION ALL SELECT concat('http://localhost:8890/Demo/suppliers/', o."SupplierID") AS o FROM "Suppliers" AS o WHERE o."City" = 'Atlanta' ''')
    
def test_shared_prop_with_class(check):
    check('''select ?o { ?o a Demo:Customers; Demo:city "Atlanta"}''',
                '''SELECT concat('http://localhost:8890/Demo/customers/', o."CustomerID") AS o FROM "Customers" AS o WHERE o."City" = 'Atlanta' ''')
    
def test_sparql_join(check):
    check('''select ?cn ?cc { { ?c a Demo:Customers; Demo:companyname ?cn . } { ?c Demo:city ?cc } }''', 
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
    
def test_sparql_join_two_tables(check):
    check('''select (COUNT(*) AS ?count) {
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

def test_orderby_limit(check):
    check('''select ?order_date {
                    ?o a Demo:Orders; Demo:orderdate ?order_date; Demo:freight ?fr.
                    } order by ?fr limit 5''',
                    '''SELECT o."OrderDate" AS order_date 
                    FROM "Orders" AS o 
                    ORDER BY o."Freight" LIMIT 5 OFFSET 0''')

def test_orderby_desc_limit_offset(check):
    check('''select ?order_date {
                    ?o a Demo:Orders; Demo:orderdate ?order_date; Demo:freight ?fr; Demo:shippeddate ?sd.
                    } order by ?fr desc(?sd) limit 5 offset 10''',
                    '''SELECT o."OrderDate" AS order_date 
                    FROM "Orders" AS o 
                    ORDER BY o."Freight", o."ShippedDate" DESC LIMIT 5 OFFSET 10''')
    
def test_blank_node(check):
    check('''select ?shid ?fr { ?sh Demo:shipperid ?shid; Demo:shippers_of_orders [ Demo:freight ?fr ]. }''',
                '''SELECT sh."ShipperID" AS shid, o."Freight" AS fr 
                FROM "Shippers" AS sh, "Orders" AS o 
                WHERE sh."ShipperID" = o."ShipVia"''')
    
def test_path(check):
    check('''select ?shid ?fr { ?sh Demo:shipperid ?shid; Demo:shippers_of_orders / Demo:freight ?fr . }''',
                '''SELECT sh."ShipperID" AS shid, o."Freight" AS fr 
                FROM "Shippers" AS sh, "Orders" AS o 
                WHERE sh."ShipperID" = o."ShipVia"''')
    
def test_const_query(check):
    check('''select (1 as ?one) {}''', 'SELECT 1 AS one')

def test_in_op(check):
    check('''select (1 in (1,2,3) as ?itsin) {}''', 'SELECT 1 IN (1, 2, 3) AS itsin')

def test_aggregate_join(check):
    check('''select ?shid (sum(?fr) as ?total_fr) { ?sh Demo:shipperid ?shid; Demo:shippers_of_orders / Demo:freight ?fr. } group by ?shid''',
                '''SELECT sh."ShipperID" AS shid, sum(o."Freight") AS total_fr 
                FROM "Shippers" AS sh, "Orders" AS o 
                WHERE sh."ShipperID" = o."ShipVia" GROUP BY sh."ShipperID"''')

def test_aggregate_join_count(check):
    check('''select ?shid (count(distinct ?city) as ?city_count) { ?sh Demo:shipperid ?shid; Demo:shippers_of_orders / Demo:shipcity ?city. } group by ?shid''',
                '''SELECT sh."ShipperID" AS shid, count(DISTINCT o."ShipCity") AS city_count 
                FROM "Shippers" AS sh, "Orders" AS o 
                WHERE sh."ShipperID" = o."ShipVia" GROUP BY sh."ShipperID"''')

def test_aggregate_join_count_star(check):
    check('''select ?shid (count(*) as ?combo_count) { ?sh Demo:shipperid ?shid; Demo:shippers_of_orders / Demo:shipcity ?city. } group by ?shid''',
                '''SELECT sh."ShipperID" AS shid, count(*) AS combo_count 
                FROM "Shippers" AS sh, "Orders" AS o 
                WHERE sh."ShipperID" = o."ShipVia" GROUP BY sh."ShipperID"''')
    
def test_regex_to_like(check):
    check('''select ?city { ?o Demo:shipcity ?city. filter regex(?city, "^.A[b-c][^d-f%*].*$", "is") }''',
                '''SELECT o."ShipCity" AS city 
                FROM "Orders" AS o 
                WHERE o."ShipCity" LIKE '_A[b-c][^d-f%*]%' ''')
    
def test_sql_func(check):
    check('''select ?city { ?o Demo:shipcity ?city. filter (sqlf:LOWER(?city) = "atlanta") }''',
                '''SELECT o."ShipCity" AS city 
                FROM "Orders" AS o 
                WHERE LOWER(o."ShipCity") = 'atlanta' ''')

def test_case(check):
    check('''SELECT (IF(4 > 3, "Yes", IF(3 < 4, "Whut", "No")) AS ?r) { }''',
                '''SELECT CASE WHEN (4 > 3) THEN 'Yes' WHEN (3 < 4) THEN 'Whut' ELSE 'No' END AS r''')
    
# For issue #2
def test_disappearing_select(check):
    check(
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

def test_join_two_selects(check):
    check('''select ?cn ?cc { 
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
    
def test_join_select_with_aggregation(check):
    check('''
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
    
def test_join_select_with_aggregation_in_subquery(check):
    check('''
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
    
def test_if(check):
    check('''SELECT (IF(4 > 3, "Yes", IF(3 < 4, "Whut", "No")) AS ?r) { }''',
                '''SELECT CASE WHEN (4 > 3) THEN 'Yes' WHEN (3 < 4) THEN 'Whut' ELSE 'No' END AS r''')
def test_arithmetic(check):
    check(
        '''select (?o_Freight * 1000 as ?grams) { ?o a Demo:Orders. ?o Demo:freight ?o_Freight. }''',
        '''SELECT o."Freight" * 1000 AS grams FROM "Orders" AS o''')

# Bug #3
def test_multiple_multiplicative_ops(check):
    check(
        '''SELECT (2 * 3 / 5 AS ?Answer) {}''',
        '''SELECT CAST((2 * 3) / 5 AS REAL) AS "Answer"''')
    
def test_multiple_additive_ops(check):
    check(
        '''SELECT (2 - 3 + 5 AS ?Answer) {}''',
        '''SELECT (2 - 3) + 5 AS "Answer"''')
    
def test_having(check):
    check('''select ?shid (sum(?fr) as ?total_fr) { 
                ?sh Demo:shipperid ?shid; Demo:shippers_of_orders ?o.
                ?o Demo:freight ?fr; Demo:orderid ?oid. 
                } group by ?shid  HAVING (COUNT(DISTINCT ?oid) > 1)''',
        '''SELECT sh."ShipperID" AS shid, sum(o."Freight") AS total_fr 
            FROM "Shippers" AS sh, "Orders" AS o 
            WHERE sh."ShipperID" = o."ShipVia" GROUP BY sh."ShipperID" HAVING count(DISTINCT o."OrderID") > 1''')
    
def test_subquery_in_aggregate(check):
    check('''SELECT (MAX(?fr) AS ?MaxFreight) {
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
    
def test_count_star(check):
    check('''SELECT (COUNT(*) AS ?ShippersWithMoreThanFiveOrders)
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
    

def test_unbound(check):
    check(
    '''SELECT ?oh_OrderID { FILTER (!BOUND(?oh_Freight)) ?oh a Demo:Orders. ?oh Demo:freight ?oh_Freight. ?oh Demo:orderid ?oh_OrderID. }''',
    '''SELECT oh."OrderID" AS "oh_OrderID" FROM "Orders" AS oh WHERE oh."Freight" IS NULL'''
    )

def test_bound(check):
    check(
    '''SELECT ?oh_OrderID { FILTER (BOUND(?oh_Freight)) ?oh a Demo:Orders. ?oh Demo:freight ?oh_Freight. ?oh Demo:orderid ?oh_OrderID. }''',
    '''SELECT oh."OrderID" AS "oh_OrderID" FROM "Orders" AS oh WHERE not(oh."Freight" IS NULL)'''
    )

def test_exists(check):
    check(
        '''select ?sid { ?s a Demo:Shippers. ?s Demo:shipperid ?sid. filter exists { ?s Demo:shippers_of_orders ?o. ?o a Demo:Orders. } }''',
        '''SELECT s."ShipperID" AS sid
        FROM "Shippers" AS s 
        WHERE EXISTS (SELECT * FROM "Orders" AS o WHERE s."ShipperID" = o."ShipVia")'''
    )

def test_simple_left_join(check):
    check(
        '''SELECT ?shid ?fr { ?s a Demo:Shippers. ?s Demo:shipperid ?shid. OPTIONAL { ?o a Demo:Orders. ?s Demo:shippers_of_orders ?o. ?o Demo:freight ?fr. }  }''',
                '''SELECT s."ShipperID" AS shid, o."Freight" AS fr 
                FROM "Shippers" AS s LEFT OUTER JOIN "Orders" AS o ON s."ShipperID" = o."ShipVia"'''
    )

def test_left_join(check):
    check('''SELECT ?shid ?fr { ?s a Demo:Shippers. OPTIONAL { ?o a Demo:Orders. ?s Demo:shippers_of_orders ?o. ?o Demo:freight ?fr. } ?s Demo:shipperid ?shid. }''',
                '''SELECT s."ShipperID" AS shid, o."Freight" AS fr 
                FROM "Shippers" AS s LEFT OUTER JOIN "Orders" AS o ON s."ShipperID" = o."ShipVia"''')        
    
def test_column_for_inverse_path(check):
    check('''SELECT ?oid ?shid { ?o a Demo:Orders; Demo:orderid ?oid; ^Demo:shippers_of_orders / Demo:shipperid ?shid }''',
                '''SELECT o."OrderID" AS oid, o."ShipVia" AS shid FROM "Orders" AS o''')

@pytest.fixture
def patched_converter_for_test_column_for_direct_path(mapping_graph:Graph, test_db:Engine) -> SQLConverter:
    #1. Find the right object map.
    om = list(mapping_graph.query( 
        '''select ?om {
            ?tm rr:logicalTable [  rr:tableName "Order Details" ];
                rr:predicateObjectMap [ rr:predicateMap [ rr:constant Demo:order_details_has_orders ] ; rr:objectMap ?om ]
        }
        '''
    ))[0][0] # type: ignore

    #2. Remove the template mapping
    mapping_graph.remove((om, rr.template, None))

    #3. Add join-based mapping instead.
    tm = list(mapping_graph.subjects(SequencePath(rr.logicalTable, rr.tableName), Literal("Orders")))[0]
    mapping_graph.add((om, rr.parentTriplesMap, tm))
    jc = BNode()
    mapping_graph.add((om, rr.joinCondition, jc))
    mapping_graph.add((jc, rr.child, Literal("OrderID")))
    mapping_graph.add((jc, rr.parent, Literal("OrderID")))
    return SQLConverter(test_db, mapping_graph)

def test_column_for_direct_path(patched_converter_for_test_column_for_direct_path:SQLConverter):
    
    # Note: graph patched in setUp
    Checker(patched_converter_for_test_column_for_direct_path)('''SELECT ?od ?oid { ?od a Demo:Order_Details; Demo:order_details_has_orders / Demo:orderid ?oid }''',
                '''SELECT concat('http://localhost:8890/Demo/order_details/', od."OrderID", '/', od."ProductID") AS od, od."OrderID" AS oid FROM "Order Details" AS od''')


def test_cast(check):
    check(
        'SELECT (xsd:decimal(42) AS ?flt42) { }',
        '''SELECT CAST(42 AS NUMERIC) AS flt42'''
        )    
    
def test_left_join_with_eq(check):
    check(
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
def test_named_subquery_expression_in_exists(check):
    check(
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
def test_two_subqueries_one_named_in_exists(check):
    check(
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

def test_select_distinct_in_subquery(check):
    check(
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
    
def test_neg(check):
    check('''SELECT (-?t0_Freight AS ?neg_freight) { ?t0 a Demo:Orders. ?t0 Demo:freight ?t0_Freight. }''',
                '''SELECT -t0."Freight" AS neg_freight FROM "Orders" AS t0''')        
    

def test_values_clause(check):
    check('''SELECT ?x ?y ?z { VALUES (?x ?y ?z) { (1 2 3) (4 5 6) (7 8 9) } }''',
                '''SELECT vals.x AS x, vals.y AS y, vals.z AS z FROM ( VALUES (1, 2, 3), (4, 5, 6), (7, 8, 9)) AS vals (x, y, z)''')

def test_values_clause_at_end(check):
    check('''SELECT ?x { ?o a Demo:Orders. ?o Demo:shipcity ?city. ?o Demo:orderid ?x } VALUES ?city { "London" "Paris" "Madrid" }''',
                '''SELECT o."OrderID" AS x FROM "Orders" AS o, ( VALUES ('London'), ('Paris'), ('Madrid')) AS vals (city) WHERE o."ShipCity" = vals.city''')

@pytest.mark.xfail(reason="Multiple VALUES clauses side by side not supported yet")
#Right now multiple VALUES clauses side by side are not supported: come out with the same name 'vals'.
def test_two_values_clauses(check):
    check('''SELECT ?x ?y { 
                    VALUES ?x { 1 2 } 
                    VALUES ?y { 3 4 } 
                }''',
                '''SELECT vals.x AS x, vals.y AS y FROM ( VALUES (1), (2)) AS vals (x), ( VALUES (3), (4)) AS vals2 (y)''')
    
@pytest.fixture
def patched_converter_for_aux_table(mapping_graph:Graph, test_db:Engine) -> SQLConverter:
    #1. Find the right object map.
    om = list(mapping_graph.query( 
        '''select ?om {
            ?tm rr:logicalTable [  rr:tableName "Orders" ];
                rr:predicateObjectMap [ rr:predicateMap [ rr:constant Demo:shipcity ] ; rr:objectMap ?om ]
        }
        '''
    ))[0][0] # type: ignore

    #2. Remove the column    mapping
    mapping_graph.remove((om, rr.column, None))

    #3. Add template mapping instead.
    mapping_graph.add((om, rr.template, Literal("http://localhost:8890/Demo/City/{ShipCity}")))


    #4. Add auxiliary triples map for cities.
    TTL = '''
@prefix rr: <http://www.w3.org/ns/r2rml#> .
@prefix Demo: <http://localhost:8890/schemas/Demo/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

    <#TriplesMapCity> a rr:TriplesMap; rr:logicalTable [ rr:tableSchema "Demo" ; rr:tableOwner "demo" ; rr:tableName "Orders" ]; 
rr:subjectMap [ rr:termtype "IRI"  ; rr:template "http://localhost:8890/Demo/City/{ShipCity}"; rr:class Demo:City; rr:graph <http://localhost:8890/Demo#> ];
rr:predicateObjectMap [ rr:predicateMap [ rr:constant rdfs:label ] ; rr:objectMap [ rr:column "ShipCity" ]; ] .'''

    #5. Change property for CategoryName to rdfs:label.
    pm = list(mapping_graph.query('''select ?pm { ?pm rr:constant Demo:categoryname }'''))[0][0]  # type: ignore
    mapping_graph.set((pm, rr.constant, RDFS.label))

    mapping_graph.parse(data=TTL, format="turtle")
    return SQLConverter(test_db, mapping_graph)

def test_aux_table_basic(patched_converter_for_aux_table:SQLConverter):
    Checker(patched_converter_for_aux_table)('''SELECT ?city ?city_name { ?city a Demo:City. ?city rdfs:label ?city_name. }''',
                '''SELECT concat('http://localhost:8890/Demo/City/', city."ShipCity") AS city, city."ShipCity" AS city_name FROM "Orders" AS city''')

def test_aux_table_link(patched_converter_for_aux_table:SQLConverter):
    Checker(patched_converter_for_aux_table)('''SELECT ?freight ?city { ?o a Demo:Orders. ?o Demo:freight ?freight. ?o Demo:shipcity ?city. }''',
                '''SELECT o."Freight" AS freight, concat('http://localhost:8890/Demo/City/', o."ShipCity") AS city FROM "Orders" AS o''')

#XXX We should not need to have the same table twice.
def test_2_classes_one_table(patched_converter_for_aux_table:SQLConverter):
    Checker(patched_converter_for_aux_table)('''SELECT ?freight ?city ?city_name{ ?o a Demo:Orders. ?o Demo:freight ?freight. ?o Demo:shipcity ?city. ?city a Demo:City. ?city rdfs:label ?city_name. }''',
                '''SELECT o."Freight" AS freight, concat('http://localhost:8890/Demo/City/', o."ShipCity") AS city, o."ShipCity" AS city_name FROM "Orders" AS o''')
    
#XXX We should not need to have the same table twice.
def test_2_classes_one_table_no_type(patched_converter_for_aux_table:SQLConverter):
    Checker(patched_converter_for_aux_table)('''SELECT ?freight ?city_name{ ?o a Demo:Orders. ?o Demo:freight ?freight. ?o Demo:shipcity ?city. ?city rdfs:label ?city_name. }''',
                '''SELECT o."Freight" AS freight, o."ShipCity" AS city_name FROM "Orders" AS o''')
        

