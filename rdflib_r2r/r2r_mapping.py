import logging
import re
from types import NoneType
import urllib.parse
from dataclasses import dataclass
import base64
import datetime
import parse
from string import Formatter

from typing import Any, Iterable, List, Tuple, Union, Dict, cast

from rdflib import Graph, URIRef, Literal, BNode
from rdflib.term import Node
from rdflib.namespace import RDF, XSD, Namespace

from sqlalchemy import MetaData, text, table, literal_column, types as sqltypes, ColumnElement, TableClause, Subquery
from sqlalchemy import or_ as sql_or, and_ as sql_and
from sqlalchemy.sql.expression import ClauseElement
from sqlalchemy.sql.selectable import NamedFromClause
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

class R2RMapping:
    graph:Graph
    baseuri: str
    spat_tmaps: Dict["Pattern",List[Node]] 
    ppat_pomaps: Dict["Pattern",List[Node]]
    opat_pomaps: Dict["Pattern",List[Node]]

    @dataclass(eq=True, order=True, frozen=True)
    class Pattern:
        tname: str
        const: str|None = None
        field: str|None = None
        parser: str|None = None
        inverse: str|None = None

    @classmethod
    def from_db(cls, db, baseuri="http://example.com/base/"):
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

        logging.warn("\n" + mg.serialize(format="turtle"))
        return cls(mg, baseuri=baseuri)

    @classmethod
    def _template_to_parser(cls, template, irisafe=None):
        # TODO: iri safe?
        # escape curly braces by doubling {{ }}
        template = re.sub("\\\\[{}]", lambda x: x.group(0)[1] * 2, template)
        template = re.sub('{"', "{", template)
        template = re.sub('"}', "}", template)
        # Replace space by double underscore...
        template = re.sub(
            "{([^{}]+)}", lambda x: x.group(0).replace(" ", "__"), template
        )
        return template

    @classmethod
    def _term_pat(cls, graph:Graph, dbtable: NamedFromClause, parent:Node, mapper:URIRef, shortcut:URIRef, obj=False) -> Iterable[Pattern]:
        """Get pattern Patterns based on term mappings

        Args:
            graph: Mapping graph
            parent: Parent mapping
            shortcut: Shortcut property
            mapper: Mapper property
            obj (bool, optional): Whether this is an object mapping. Defaults to False.

        Yields:
            R2RMapping.Pattern: Node n3 pattern
        """
        if graph.value(parent, shortcut):
            # constant shortcut properties
            for const in graph.objects(parent,shortcut):
                yield cls.Pattern(const=toPython(const), tname=dbtable.name)
        elif graph.value(parent, mapper):
            for tmap in graph.objects(parent, mapper):
                if graph.value(tmap, rr.constant):
                    # constant value
                    for const in graph.objects(tmap, rr.constant):
                        yield cls.Pattern(const=toPython(const), tname=dbtable.name)
                else:
                    # Inverse Expression
                    inverse = None
                    for inv_exp in graph.objects(tmap, rr.inverseExpression):
                        inverse = cls._template_to_parser(inv_exp)

                    termtype = graph.value(tmap, rr.termType) or rr.IRI
                    if graph.value(tmap, rr.column):
                        col = graph.value(tmap, rr.column)
                        assert col
                        col = re.sub('(^"|"$)', "", str(col))
                        yield cls.Pattern(field=col, inverse=inverse, tname=dbtable.name)
                    elif graph.value(tmap, rr.template):
                        template = graph.value(tmap, rr.template)
                        parser = cls._template_to_parser(
                            template, irisafe=(termtype == rr.IRI)
                        )
                        yield cls.Pattern(parser=parser, inverse=inverse, tname=dbtable.name)
                    elif graph.value(tmap, rr.parentTriplesMap):
                        # referencing object map
                        ref = graph.value(tmap, rr.parentTriplesMap)
                        assert ref
                        rt = _get_table(graph, ref)
                        rt = rt.alias(f"{rt.name}_ref")
                        rs = cls._term_pat(graph, rt, ref, rr.subjectMap, rr.subject)
                        yield from rs
                    else:
                        t = dbtable.name
                        parser = cls._template_to_parser(f"{t}#{{rowid}}")
                        yield cls.Pattern(parser=parser, inverse=inverse, tname=t)

    def __init__(self, g:Graph, baseuri="http://example.com/base/"):
        self.graph = g
        self.baseuri = baseuri

        # Track triple maps per node pattern Pattern
        self.spat_tmaps, self.ppat_pomaps, self.opat_pomaps = {}, {}, {}
        for tmap in g.subjects(RDF.type, rr.TriplesMap):
            t = _get_table(g, tmap)
            for s_pat in self._term_pat(g, t, tmap, rr.subjectMap, rr.subject):
                self.spat_tmaps.setdefault(s_pat, []).append(tmap)
            for po_i, pomap in enumerate(g.objects(tmap, rr.predicateObjectMap)):
                for p_pat in self._term_pat(g, t, pomap, rr.predicateMap, rr.predicate):
                    self.ppat_pomaps.setdefault(p_pat, []).append(pomap)
                for o_pat in self._term_pat(g, t, pomap, rr.objectMap, rr.object, True):
                    self.opat_pomaps.setdefault(o_pat, []).append(pomap)

    def inverse_condition(self, inverse_expr: str, field_values: Dict) -> ColumnElement[bool]:
        col_replace = {
            col: f'"{col}"'
            for _, col, _, _ in Formatter().parse(inverse_expr)
            if (col != None) and (col not in field_values)
        }
        param_replace = {field: ":" + field for field in field_values}
        t = text(inverse_expr.format(**col_replace, **param_replace))
        return cast(ColumnElement[bool], t.bindparams(**field_values))

    def get_node_filter(self, node:Node|None, pat_maps:Dict[Pattern,List[Node]]) -> Dict[Node|NoneType, List[ColumnElement[bool]]]:
        
        # A pattern may be used in multiple places, so use sql_or
        # (but what happens if the table name is out of scope ...? )
        map_conditions:Dict[Node,List[ColumnElement[bool]]] = {}
        if node is not None:
            for pat, maps in pat_maps.items():
                if pat.const:
                    if toPython(node) == pat.const:
                        for m in maps:
                            # No condition, select all
                            map_conditions.setdefault(m, []).append(TRUE_ELT)
                elif pat.field:
                    val = toPython(node)
                    if isinstance(val, str):
                        # not super sure of this:
                        val = iri_unsafe(val).replace(self.baseuri, "").strip()
                    elif isinstance(val, datetime.date):
                        val = str(val)
                    elif isinstance(val, bytes):
                        val = text(f"x'{base64.b16encode(val).decode()}'")
                    
                    if pat.inverse:
                        fields = {pat.field: val}
                        where = self.inverse_condition(f'"{pat.tname}".{pat.inverse}', fields)
                    else:
                        where = literal_column(f'"{pat.tname}"."{pat.field}"').__eq__(val) # Not using == because of type issues
                    for m in maps:
                        if not (where is True):
                            map_conditions.setdefault(m, []).append(where)
                elif pat.parser:
                    parser = parse.compile(pat.parser)
                    val = str(toPython(node)).replace(self.baseuri, "").strip()

                    parsed_val = parser.parse(val)
                    if parsed_val:
                        assert isinstance(parsed_val, Result)
                        fields = parsed_val.named
                        if pat.inverse:
                            where = self.inverse_condition(pat.inverse, fields)
                            # where = literal_column(f'"{pat.tname}".{where}')
                        else:
                            # A template pattern may have multiple fields;
                            #   all must match, so use sql_and
                            where_clauses = []
                            for key, val in fields.items():
                                key = f'"{pat.tname}"."{key.replace("__", " ")}"'
                                if pat.parser.startswith("data:"):
                                    val = text(f"x'{val}'")
                                else:
                                    val = iri_unsafe(val)
                                clause = literal_column(key) == val
                                where_clauses.append(clause)
                            where = sql_and(*where_clauses) if where_clauses else TRUE_ELT
                        for m in maps:
                            if not (where is True):
                                map_conditions.setdefault(m, []).append(where)
            return {
                m: ([sql_or(*wheres)] if wheres and not any(w is TRUE_ELT for w in wheres) else [])
                for m, wheres in map_conditions.items()
            }
        else:
            return {None: []}