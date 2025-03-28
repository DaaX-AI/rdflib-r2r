from typing import Any, Generator, Mapping, Optional
from rdflib.plugins.sparql.sparql import Query
from rdflib.query import Result
from rdflib.term import Identifier
from rdflib_r2r.conversion_utils import ImpossibleQueryException, sql_pretty
from rdflib_r2r.r2r_mapping import iri_safe
from rdflib_r2r.sql_converter import SQLConverter
from rdflib.plugins.stores.sparqlstore import SPARQLStore


from rdflib import BNode, Graph, Literal, URIRef, Variable
from rdflib.namespace import XSD
from rdflib.plugins.sparql.parserutils import CompValue
from rdflib.store import Store
from rdflib.util import from_n3
from sqlalchemy.engine import Engine
from sqlalchemy.sql.elements import NamedColumn
from rdflib_r2r.conversion_utils import get_template_expansion_info


import base64
import logging
import re
from abc import ABC

from rdflib_r2r.types import SQLQuery


class R2RStore(SPARQLStore):

    converter: SQLConverter
    """
    Args:
        db: SQLAlchemy engine.
    """
    def __init__(
        self,
        db: Engine,
        mapping_graph: Graph,
        base: str = "http://example.com/base/",
        configuration=None,
        identifier=None,
    ):
        super().__init__(
            configuration=configuration, identifier=identifier
        )
        self.db = db
        self.mapping_graph = mapping_graph
        self.base = base
        self._current_project = None
        self.converter = SQLConverter(db, mapping_graph)
        assert self.db

    def _query(self, query:str, default_graph: Optional[str] = None) -> Result:
        if default_graph:
            logging.warning(f"def graph: {default_graph}")
        return self.query(query)

    def query(self, query: Query | str, initNs: Mapping[str, Any] | None = None, 
              initBindings: Mapping[str, Identifier] | None = None, queryGraph: str | None = None, 
              DEBUG: bool = False) -> Result:
        if initBindings:
            v = list(initBindings)
            if isinstance(query, str):
                query += "\nVALUES ( %s )\n{ ( %s ) }\n" % (
                    " ".join("?" + str(x) for x in v),
                    " ".join(self.node_to_sparql(initBindings[x]) for x in v),
                )
            else:
                # Build the values into the right place in the query algebra
                vals = CompValue("values", res=[{ Variable(k):v  for k,v in initBindings.items() }])
                cv = query.algebra
                while cv.p.name in {"Project", "OrderBy", "Extend"}:
                    cv = cv.p
                cv.p = CompValue("Join", p1=cv.p, p2=vals)

        # XXX Figure these out later
        #if queryGraph:
        #    raise NotImplementedError
        query_obj = query if isinstance(query, Query) else self.converter.parse_sparql_query(query, base=self.base, initNs=initNs)
        vars = query_obj.algebra["PV"]
        try:
            sql = self.converter.get_sql_query_object(query_obj)
        except ImpossibleQueryException:
            sql = None
        except NotImplementedError as e:
            logging.warning(f"SQL translation not implemented: {e}; query:{query}")
            raise
        bindings = self.exec(sql) if sql is not None else []
        r = Result("SELECT")
        r.vars = vars
        r.bindings = bindings
        return r


    def __len__(self, context=None) -> int:
        """The number of RDF triples in the DB mapping."""
        raise NotImplementedError

    @property
    def nb_subjects(self) -> int:
        """The number of subjects in the DB mapping."""
        raise NotImplementedError

    @property
    def nb_predicates(self) -> int:
        """The number of predicates in the DB mapping."""
        raise NotImplementedError

    @property
    def nb_objects(self) -> int:
        """The number of objects in the DB mapping."""
        raise NotImplementedError

    @property
    def nb_shared(self) -> int:
        """The number of shared subject-object in the DB mapping."""
        raise NotImplementedError

    def _iri_encode(self, iri_n3) -> URIRef:
        iri = iri_n3[1:-1]
        uri = re.sub("<ENCODE>(.+?)</ENCODE>", lambda x: iri_safe(x.group(1)), iri)
        return URIRef(uri, base=self.base)

    def make_node(self, val) -> Identifier|None:
        if val is None:
            return None
        isstr = isinstance(val, str)
        if (not isstr) or (val[0] not in '"<_'):
            if type(val) == bytes:
                return Literal(
                    base64.b16encode(val),
                    datatype=XSD.hexBinary,
                )
            else:
                # TODO: actually figure out the rules for this
                # if type(val) == float:
                    # if math.isclose(val, round(val, 2)):
                    #     val = Decimal(val)
                return Literal(val)
        elif val.startswith("<"):
            return self._iri_encode(val)
        elif val == "_:":
            return BNode()
        elif val.startswith("_:"):
            return BNode(val[2:])
        else:
            raise ValueError(f"Unexpected value: {val}")

    def exec(self, query:SQLQuery) -> Generator[Mapping[Variable, Identifier], None, None]:
        with self.db.connect() as conn:
            results = conn.execute(query)
            rows = list(results)
            keys = [Variable(v) for v in results.keys()]
            iris = set()
            for c in query.exported_columns:
                if isinstance(c, NamedColumn):
                    ei = get_template_expansion_info(c)
                    if ei and ei.is_iri:
                        iris.add(c.name)

            first = True
            for vals in rows:
                if first:
                    first = False
                bindings = {}
                for var, val in zip(keys, vals):
                    bindings[var] = URIRef(val) if str(var) in iris else self.make_node(val)
                yield bindings



