import logging
import urllib.parse
import base64

from typing import Any

from rdflib import Graph, URIRef, Literal, BNode
from rdflib.term import Node
from rdflib.namespace import RDF, XSD, Namespace

from sqlalchemy import Engine, MetaData, text, table, types as sqltypes
from sqlalchemy.sql.elements import True_

from rdflib_r2r.sql_view import view2obj
from parse import Result
rr = Namespace("http://www.w3.org/ns/r2rml#")



def iri_safe(v):
    return urllib.parse.quote(v, safe="")


def iri_unsafe(v):
    return urllib.parse.unquote(v)


def _get_table(graph:Graph, tmap:Node):
    logtable = graph.value(tmap, rr.logicalTable)
    if graph.value(logtable, rr.tableName):
        tname = str(graph.value(logtable, rr.tableName))
        return table(tname.strip('"'))
    else:
        tname = f'"View_{base64.b32encode(str(tmap).encode()).decode()}"'
        sqlquery = str(graph.value(logtable, rr.sqlQuery)).strip().strip(";")

        # TODO: parse views to SQLAlchemy objects to get column types
        view2obj(sqlquery)

        return text(sqlquery).columns().subquery(tname.strip('"'))

def toPython(node:Node) -> Any:
    if isinstance(node, Literal) or isinstance(node, URIRef):
        return node.toPython()
    elif isinstance(node, BNode):
        return str(node)
    
TRUE_ELT = True_._singleton

def mapping_from_db(db:Engine, baseuri="http://example.com/base/") -> Graph:
    """Create RDB2RDF Direct Mapping

    See also: https://www.w3.org/TR/rdb-direct-mapping/

    Args:
        db (sqlalchemy.Engine): Database
        baseuri (str, optional): Base URI. Defaults to "http://example.com/base/".

    Returns:
        R2RMapping
    """
    base = Namespace(baseuri)
    mg = Graph(base=baseuri)

    with db.connect() as conn:
        metadata = MetaData()
        metadata.reflect(conn)
        
        tmaps: dict[str,Node] = {}
        assert metadata.tables
        for tablename, table in metadata.tables.items():
            tmap = tmaps.setdefault(tablename, BNode())
            mg.add((tmap, RDF.type, rr.TriplesMap))
            logtable = BNode()
            mg.add((tmap, rr.logicalTable, logtable))
            mg.add((logtable, rr.tableName, Literal(f'"{tablename}"')))

            s_map = BNode()
            mg.add((tmap, rr.subjectMap, s_map))
            mg.add((s_map, rr["class"], base[iri_safe(tablename)]))

            # TEMPORARY: duckdb hack
            # duckdb returns the wrong primary keys!
            # see https://github.com/Mause/duckdb_engine/issues/594
            pk = db.dialect.get_pk_constraint(conn, tablename, schema="main")
            logging.warn(f'table:{list(table.primary_key)}')
            logging.warn(f'get_pk_constraint:{pk}')
            if pk and any(pk.values()):
                if pk['name']:
                    # for duckdb, the "constrained_columns" is wrong
                    # so we extract the key column names from the "name" field
                    keys = pk["name"].partition("KEY")[-1][1:-1].split(',')
                    primary_keys = [k.strip() for k in keys]
                else:
                    # sqlite doesn't set the "name" field
                    primary_keys = pk["constrained_columns"]
            else:
                primary_keys = []

            if primary_keys:
                parts = ['%s={"%s"}' % (iri_safe(c), c) for c in primary_keys]
                template = iri_safe(tablename) + "/" + ";".join(parts)
                mg.add((s_map, rr.template, Literal(template)))
                mg.add((s_map, rr.termType, rr.IRI))
            else:
                mg.add((s_map, rr.termType, rr.BlankNode))

            for column in db.dialect.get_columns(conn, tablename, schema="main"):
                colname = column["name"]
                coltype = column["type"]
                # Add a predicate-object map per column
                pomap = BNode()
                mg.add((tmap, rr.predicateObjectMap, pomap))
                pname = f"{iri_safe(tablename)}#{iri_safe(colname)}"
                mg.add((pomap, rr.predicate, base[pname]))

                o_map = BNode()
                mg.add((pomap, rr.objectMap, o_map))
                mg.add((o_map, rr.column, Literal(f'"{colname}"')))
                if isinstance(coltype, sqltypes.Integer):
                    mg.add((o_map, rr.datatype, XSD.integer))

            foreign_keys = db.dialect.get_foreign_keys(
                conn, tablename, schema="main"
            )
            for fk in foreign_keys:
                # Add another predicate-object map for every foreign key
                pomap = BNode()
                mg.add((tmap, rr.predicateObjectMap, pomap))
                parts = [iri_safe(part) for part in fk["constrained_columns"]]
                pname = f"{iri_safe(tablename)}#ref-{';'.join(parts)}"
                mg.add((pomap, rr.predicate, base[pname]))

                o_map = BNode()
                mg.add((pomap, rr.objectMap, o_map))
                reftable = fk["referred_table"]
                refmap = tmaps.setdefault(reftable, BNode())
                mg.add((o_map, rr.parentTriplesMap, refmap))

                colpairs = zip(fk["constrained_columns"], fk["referred_columns"])
                for colname, refcol in colpairs:
                    join = BNode()
                    mg.add((o_map, rr.joinCondition, join))
                    mg.add((join, rr.child, Literal(f'"{colname}"')))
                    mg.add((join, rr.parent, Literal(f'"{refcol}"')))

    return mg

