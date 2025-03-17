from rdflib_r2r.conversion_utils import sql_pretty
from rdflib_r2r.r2r_mapping import iri_safe
from rdflib_r2r.sql_converter import SQLConverter


from rdflib import BNode, Graph, Literal, URIRef, Variable
from rdflib.namespace import XSD
from rdflib.plugins.sparql.parserutils import CompValue
from rdflib.store import Store
from rdflib.util import from_n3
from sqlalchemy.engine import Engine


import base64
import logging
import re
from abc import ABC


class R2RStore(Store, ABC):

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
        super(R2RStore, self).__init__(
            configuration=configuration, identifier=identifier
        )
        self.db = db
        self.mapping_graph = mapping_graph
        self.base = base
        self._current_project = None
        self.converter = SQLConverter(db, mapping_graph)
        assert self.db

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

    def make_node(self, val):
        isstr = isinstance(val, str)
        if val is None:
            return None
        elif (not isstr) or (val[0] not in '"<_'):
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
            return from_n3(val)
        else:
            return from_n3(val)

    def exec(self, query):
        with self.db.connect() as conn:
            logging.warning("Executing:\n" + sql_pretty(query))
            # raise Exception
            results = conn.execute(query)
            rows = list(results)
            keys = [Variable(v) for v in results.keys()]
            logging.warning(f"Got {len(rows)} rows of {keys}")
            first = True
            for vals in rows:
                if first:
                    logging.warning(f"First row: {vals}")
                    first = False
                yield dict(zip(keys, [self.make_node(v) for v in vals]))


    def evalPart(self, part:CompValue):
        query = self.converter.queryPart(part)
        return self.exec(query)

    def create(self, configuration):
        raise TypeError("The DB mapping is read only!")

    def destroy(self, configuration):
        raise TypeError("The DB mapping is read only!")

    def commit(self):
        raise TypeError("The DB mapping is read only!")

    def rollback(self):
        raise TypeError("The DB mapping is read only!")

    def add(self, triple, context=None, quoted=False):
        raise TypeError("The DB mapping is read only!")

    def addN(self, quads):
        raise TypeError("The DB mapping is read only!")

    def remove(self, triple, context=None):
        raise TypeError("The DB mapping is read only!")