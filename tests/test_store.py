import unittest

from rdflib import Graph, Literal, URIRef
from sqlalchemy import MetaData, column, literal, select, table, text, func as sqlfunc

from rdflib_r2r.r2r_store import R2RStore
from tests.test_sql_converter import BaseSQLConvertingTest, DEMO_NS



class TestStore(BaseSQLConvertingTest):

    store:R2RStore

    def setUp(self) -> None:
        super().setUp()
        self.store = R2RStore(self.db, self.mapping_graph)
        self.g = Graph(self.store)

    def test_order_by_id(self):
        self.assertEqual([(URIRef('http://localhost:8890/Demo/orders/10248'),DEMO_NS.orderid, Literal(10248))],
            list(self.g.triples((None, DEMO_NS.orderid, Literal(10248)))))

    # ASK not implemented
    @unittest.expectedFailure
    def test_confirm_product_name(self):
        self.assertEqual([(URIRef('http://localhost:8890/Demo/products/1'), DEMO_NS.productname, Literal('Chai'))],
            list(self.g.triples((URIRef('http://localhost:8890/Demo/products/1'), DEMO_NS.productname, Literal('Chai')))))        
    
    def test_get_product_name(self):
        self.assertEqual([(URIRef('http://localhost:8890/Demo/products/1'), DEMO_NS.productname, Literal('Chai'))],
            list(self.g.triples((URIRef('http://localhost:8890/Demo/products/1'), DEMO_NS.productname, None))))
        
    def test_get_products_a2z(self):
        self.assertEqual([(Literal('Alice Mutton'),), (Literal('Aniseed Syrup'),)],
            list(self.g.query(f'SELECT ?pn WHERE {{ ?product a Demo:Products; Demo:productname ?pn }} ORDER BY ?pn LIMIT 2', initNs=self.ns_map))
        )
