"""
The mapped SQL database as an RDF graph store object

.. note::
   TODO:
   
   * delay the UNION creation longer, so that we can use colforms to filter them

    * This could be complicated because you'd need to explode the BGPs 
      because each triple pattern becomes a set of selects, which you'd need to join



"""
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from os import linesep
from typing import Any, Generator, Generic, Iterable, Mapping, Optional, List, Dict, NamedTuple, cast, TypeVar, overload, Sequence
import logging
import base64
import re

from rdflib import URIRef, Literal, BNode, Variable
from rdflib.namespace import XSD, Namespace
from rdflib.store import Store
from rdflib.util import from_n3
from rdflib.term import Node
from rdflib.plugins.sparql.parserutils import CompValue
import sqlalchemy
import sqlalchemy.sql.operators
from sqlalchemy import ColumnElement, CompoundSelect, Label, except_, literal, select, null, literal_column
from sqlalchemy import union_all, or_ as sql_or, and_ as sql_and
from sqlalchemy import types as sqltypes, func as sqlfunc
from sqlalchemy.sql.expression import GenerativeSelect, Select
from sqlalchemy.sql.selectable import _CompoundSelectKeyword
from sqlalchemy.engine import Engine, Connection

from rdflib_r2r.expr_template import NULL_SUBFORM, ExpressionTemplate, SubForm
from rdflib_r2r.types import SQLQuery, Triple, BGP
from rdflib_r2r.r2r_mapping import R2RMapping, iri_safe, toPython

@dataclass
class SelectVarSubForm:
    select: SQLQuery #: 
    #: A map of RDF variables to the subforms that generate them from SQL expressions
    varsubforms: Dict[Variable, SubForm]

    def as_tuple(self) -> tuple[SQLQuery, Dict[Variable, SubForm]]:
        return self.select, self.varsubforms

class SparqlNotImplementedError(NotImplementedError):
    pass

rr = Namespace("http://www.w3.org/ns/r2rml#")

# https://www.w3.org/2001/sw/rdb2rdf/wiki/Mapping_SQL_datatypes_to_XML_Schema_datatypes

XSDToSQL = {
    XSD.time: sqltypes.Time(),
    XSD.date: sqltypes.Date(),
    XSD.gYear: sqltypes.Integer(),
    XSD.gYearMonth: None,
    XSD.dateTime: None,
    XSD.duration: sqltypes.Interval(),
    XSD.dayTimeDuration: sqltypes.Interval(),
    XSD.yearMonthDuration: sqltypes.Interval(),
    XSD.hexBinary: None,
    XSD.string: sqltypes.String(),
    XSD.normalizedString: None,
    XSD.token: None,
    XSD.language: None,
    XSD.boolean: sqltypes.Boolean(),
    XSD.decimal: sqltypes.Numeric(),
    XSD.integer: sqltypes.Integer(),
    XSD.nonPositiveInteger: sqltypes.Integer(),
    XSD.long: sqltypes.Integer(),
    XSD.nonNegativeInteger: sqltypes.Integer(),
    XSD.negativeInteger: sqltypes.Integer(),
    XSD.int: sqltypes.Integer(),
    XSD.unsignedLong: sqltypes.Integer(),
    XSD.positiveInteger: sqltypes.Integer(),
    XSD.short: sqltypes.Integer(),
    XSD.unsignedInt: sqltypes.Integer(),
    XSD.byte: sqltypes.Integer(),
    XSD.unsignedShort: sqltypes.Integer(),
    XSD.unsignedByte: sqltypes.Integer(),
    XSD.float: sqltypes.Float(),
    XSD.double: sqltypes.Float(),
    XSD.base64Binary: None,
    XSD.anyURI: None,
}


def sql_pretty(query:SQLQuery):
    import sqlparse

    qstr = str(query.compile(compile_kwargs={"literal_binds": True}))
    return sqlparse.format(qstr, reindent=True, keyword_case="upper")

def get_named_columns(query:Select) -> Dict[str,ColumnElement]:
    return { col.name: col for col in query.exported_columns if isinstance(col, Label)}

def results_union(queries:Sequence[SQLQuery]) -> SQLQuery:

    if len(queries) == 1:
        return queries[0]

    selects: List[Select] = []
    def add_selects(q:SQLQuery):
        if isinstance(q,Select):
            selects.append(q)
        elif q.keyword == _CompoundSelectKeyword.UNION_ALL:
            for sq in q.selects:
                if isinstance(sq, SQLQuery):
                    add_selects(sq)
                else:
                    raise ValueError(f"Unexpected sub-select: {sq}")

    for q in queries:
        add_selects(q)

    column_lists:List[List[ColumnElement]] = []
    for _ in range(len(selects)):
        column_lists.append([])

    done: set[str] = set()
    named_columns = [ get_named_columns(q) for q in selects]

    for ncs in named_columns:
        for vn in ncs:
            if vn not in done:
                done.add(vn)
                for i in range(len(selects)):
                    e = named_columns[i].get(vn,None)
                    if e is None:
                        e = null().label(vn)
                    column_lists[i].append(e)

    extended_queries = [ q.with_only_columns(*cols) for q, cols in zip(selects,column_lists) ]
    return union_all(*extended_queries)

def project_query(query:SQLQuery, names:Sequence[str]):
    actual_names = [ec.name if isinstance(ec, Label) else None for ec in query.exported_columns ]
    if actual_names == names:
        return query
    
    query = as_select(query)
    named_query_cols = get_named_columns(query)
    cols = []
    for vn in names:
        nc = named_query_cols.get(vn,None)
        cols.append(nc if nc is not None else null().label(vn))

    return query.with_only_columns(*cols)


def as_select(query:SQLQuery) ->Select:
    if isinstance(query, CompoundSelect):
        sq = query.subquery()
        query = select(*[col.label(col.key) for col in sq.exported_columns])
    return query

def equal(*expressions, eq=True) -> Generator[ColumnElement,None,None]:
    if expressions:
        e0, *es = expressions
        for e in es:
            yield (e0 == e) if eq else (e0 != e) #TODO: disassemble templates


def op(opstr:str, cf1:ColumnElement, cf2:ColumnElement) -> ColumnElement:
    if opstr in ["=", "=="]:
        return sql_and(*equal(cf1, cf2))
    elif opstr in ["!=", "<>"]:
        return sql_or(*equal(cf1, cf2, eq=False))
    elif opstr == "/":
        op = sqlalchemy.sql.operators.custom_op(opstr, is_comparison=True)
        return sqlfunc.cast(op(cf1, cf2), sqltypes.REAL)
    else:
        op = sqlalchemy.sql.operators.custom_op(opstr, is_comparison=True)
        # TODO: fancy type casting
        return op(cf1, cf2)

class R2RStore(Store, ABC):
    """
    Args:
        db: SQLAlchemy engine.
    """

    def __init__(
        self,
        db: Engine,
        mapping: Optional[R2RMapping] = None,
        base: str = "http://example.com/base/",
        configuration=None,
        identifier=None,
    ):
        super(R2RStore, self).__init__(
            configuration=configuration, identifier=identifier
        )
        self.db = db
        self.mapping = mapping or R2RMapping.from_db(db)
        self.base = base
        assert self.db
        assert self.mapping

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

    @abstractmethod
    def queryBGP(self, bgp: BGP) -> SQLQuery:
        ...

 
    def queryExpr(self, expr, var_cf:dict[str, ColumnElement]) -> ColumnElement:
        # TODO: this all could get really complicated with expression types...
        agg_funcs = {
            "Aggregate_Sample": lambda x: x,
            "Aggregate_Count": sqlfunc.count,
            "Aggregate_Sum": sqlfunc.sum,
            "Aggregate_Avg": sqlfunc.avg,
            "Aggregate_Min": sqlfunc.min,
            "Aggregate_Max": sqlfunc.max,
            "Aggregate_GroupConcat": sqlfunc.group_concat_node,
        }
        if hasattr(expr, "name") and (expr.name in agg_funcs):
            sub = self.queryExpr(expr.vars, var_cf)
            if expr.name == "Aggregate_Sample":
                return sub
            func = agg_funcs[expr.name]
            if (len(sub.cols) == 1) and (expr.name == "Aggregate_Count"):
                # Count queries don't need full node expression
                return func(sub.cols[0])
            else:
                return func(sub.expr())

        if hasattr(expr, "name") and (expr.name == "RelationalExpression"):
            a = self.queryExpr(expr.expr, var_cf)
            b = self.queryExpr(expr.other, var_cf)
            return op(expr.op, a, b)

        math_expr_names = ["MultiplicativeExpression", "AdditiveExpression"]
        if hasattr(expr, "name") and (expr.name in math_expr_names):
            # TODO: ternary ops?
            a = self.queryExpr(expr.expr, var_cf)
            for other in expr.other:
                b = self.queryExpr(other, var_cf)
                return op(expr.op[0], a, b)

        if hasattr(expr, "name") and (expr.name == "ConditionalAndExpression"):
            exprs = [self.queryExpr(e, var_cf) for e in [expr.expr] + expr.other]
            return sql_and(*exprs)

        if hasattr(expr, "name") and (expr.name == "ConditionalOrExpression"):
            exprs = [self.queryExpr(e, var_cf) for e in [expr.expr] + expr.other]
            return sql_or(*exprs)

        if hasattr(expr, "name") and (expr.name == "Function"):
            # TODO: it would be super cool to do UDFs here
            if expr.iri in XSDToSQL:
                for e in expr.expr:
                    cf = self.queryExpr(e, var_cf)
                    return sqlfunc.cast(cf.expr(), XSDToSQL[expr.iri])

        if hasattr(expr, "name") and (expr.name == "UnaryNot"):
            cf = self.queryExpr(expr.expr, var_cf)
            return sqlalchemy.not_(cf) 
        
        if hasattr(expr, "name") and (expr.name == "Builtin_BOUND"):
            cf = self.queryExpr(expr.arg, var_cf)
            return sqlfunc.not_(cf.is_(None))

        if isinstance(expr, Variable):
            result = var_cf.get(str(expr),None)
            if result is None:
                return null()
            return result
        if isinstance(expr, URIRef):
            return literal(expr.n3())
        if isinstance(expr, Literal):
            return literal(expr.toPython()) 
        if isinstance(expr, str):
            return literal(toPython(cast(Node,from_n3(expr))))

        e = f'Expr not implemented: {getattr(expr, "name", None).__repr__()} {expr}'
        raise SparqlNotImplementedError(e)

    def queryFilter(self, part:CompValue) -> SQLQuery:
        part_query = self.queryPart(part.p)
        part_query = as_select(part_query)
        named_cols = get_named_columns(part_query)

        if getattr(part.expr, "name", None) == "Builtin_NOTEXISTS":
            # This is weird, but I guess that's how it is
            query2 = self.queryPart(part.expr.graph)

            var_colforms = {}
            cols1 = list(part_query.exported_columns)
            for v, sf1 in var_subform.items():
                var_colforms.setdefault(v, []).append(ExpressionTemplate.from_subform(cols1, *sf1))
            cols2 = list(query2.exported_columns)
            for v, sf2 in var_subform2.items():
                var_colforms.setdefault(v, []).append(ExpressionTemplate.from_subform(cols2, *sf2))

            where = [eq for cs in var_colforms.values() for eq in ExpressionTemplate.equal(*cs)]
            return part_query.filter(~as_select(query2).where(*where).exists())

        clause = self.queryExpr(part.expr, named_cols)

        # Filter should be HAVING for aggregates
        if part.p.name == "AggregateJoin":
            return part_query.having(clause)
        else:
            return part_query.where(clause)

    def queryJoin(self, part) -> SQLQuery:
        query1 = self.queryPart(part.p1)
        query2 = self.queryPart(part.p2)
        if not query1.c:
            return SelectVarSubForm(query2, var_subform2)
        if not query2.c:
            return SelectVarSubForm(query1, var_subform1)

        var_colforms = {}
        cols1 = list(query1.c)
        for v, sf1 in var_subform1.items():
            var_colforms.setdefault(v, []).append(ExpressionTemplate.from_subform(cols1, *sf1))
        cols2 = list(query2.c)
        for v, sf2 in var_subform2.items():
            var_colforms.setdefault(v, []).append(ExpressionTemplate.from_subform(cols2, *sf2))

        colforms = [cfs[0] for cfs in var_colforms.values()]
        subforms, allcols = ExpressionTemplate.to_subforms_columns(*colforms)
        where = [eq for cs in var_colforms.values() for eq in ExpressionTemplate.equal(*cs)]
        return SelectVarSubForm(select(*allcols).where(*where), dict(zip(var_colforms, subforms)))

    def queryAggregateJoin(self, agg) -> SQLQuery:
        # Assume agg.p is always a Group
        group_expr, group_part = agg.p.expr, agg.p.p
        part_query, var_subform = self.queryPart(group_part)
        cols = part_query.c
        var_cf = {v: ExpressionTemplate.from_subform(cols, *sf) for v, sf in var_subform.items()}

        # Get aggregate column expressions
        var_agg = {a.res: self.queryExpr(a, var_cf) for a in agg.A}
        groups = [
            c
            for e in (group_expr or [])
            for c in self.queryExpr(e, var_cf).cols
            if type(c) != str
        ]

        subforms, allcols = ExpressionTemplate.to_subforms_columns(*var_agg.values())
        query = select(*allcols).group_by(*groups)
        return SelectVarSubForm(query, dict(zip(var_agg, subforms)))

    def queryExtend(self, part) -> SQLQuery:
        part_query = self.queryPart(part.p)
        part_query = as_select(part_query)

        cf = self.queryExpr(part.expr, get_named_columns(part_query))
        return part_query.with_only_columns(*part_query.exported_columns, cf.label(str(part.var)))

    def queryProject(self, part:CompValue) -> SQLQuery:
        part_query = self.queryPart(part.p)
        expected_names = [str(v) for v in part.PV]
        return project_query(part_query, expected_names)
    
    def queryOrderBy(self, part) -> SQLQuery:
        part_query = self.queryPart(part.p)
        cols = list(part_query.exported_columns)
        var_cf = {v: ExpressionTemplate.from_subform(cols, *sf) for v, sf in var_subform.items()}

        ordering = []
        for e in part.expr:
            expr_cf = self.queryExpr(e.expr, var_cf)
            if expr_cf.form[0] != '<':
                for col in expr_cf.cols:
                    if e.order == "DESC":
                        col = sqlalchemy.desc(col)
                    ordering.append(col)
            else:
                ordering.append(expr_cf.expr())

        return SelectVarSubForm(part_query.order_by(*ordering), var_subform)

    def queryUnion(self, part) -> SQLQuery:
        return results_union([self.queryPart(part.p1), self.queryPart(part.p2)])

    def querySlice(self, part) -> SQLQuery:
        query = self.queryPart(part.p)
        if part.start:
            query = query.offset(part.start)
        if part.length:
            query = query.limit(part.length)
        return query

    def queryLeftJoin(self, part) -> SQLQuery:

        query1, var_subform1 = self.queryPart(part.p1)
        query2, var_subform2 = self.queryPart(part.p2)
        if not query1.c:
            return SelectVarSubForm(query2, var_subform2)
        if not query2.c:
            return SelectVarSubForm(query1, var_subform1)

        var_colforms:dict[Variable,List[ExpressionTemplate]] = {}
        allcols1, allcols2 = [], []
        cols1 = list(query1.c)
        for v, sf1 in var_subform1.items():
            cf = ExpressionTemplate.from_subform(cols1, *sf1)
            var_colforms.setdefault(v, []).append(cf)
            allcols1.append(cf.expr().label(str(v)))
        
        query2 = query2.subquery()
        cols2 = list(query2.c)
        for v, sf2 in var_subform2.items():
            cf = ExpressionTemplate.from_subform(cols2, *sf2)
            var_colforms.setdefault(v, []).append(cf)
            allcols2.append(cf.expr().label(str(v)))

        where = [eq for cs in var_colforms.values() for eq in ExpressionTemplate.equal(*cs)]

        outer = select(*query1.c, *query2.c).outerjoin(
            query2, 
            onclause=sql_and(*where)
        )
        varcols = [literal_column(str(v)) for v in var_colforms]
        query = select(*varcols).select_from(outer.subquery())

        return SelectVarSubForm(query, {v: SubForm([i], (None,)) for i,v in enumerate(var_colforms)})

    def queryPart(self, part:CompValue) -> SQLQuery:
        if part.name == "BGP":
            return self.queryBGP(part.triples)
        if part.name == "Filter":
            return self.queryFilter(part)
        if part.name == "Extend":
            return self.queryExtend(part)
        if part.name == "Project":
            return self.queryProject(part)
        if part.name == "Join":
            return self.queryJoin(part)
        if part.name == "AggregateJoin":
            return self.queryAggregateJoin(part)
        if part.name == "ToMultiSet":
            # no idea what this should do
            return self.queryPart(part.p)
        if part.name == "Minus":
            q1 = self.queryPart(part.p1)
            q2 = self.queryPart(part.p2)
            return except_(q1,q2)
        if part.name == "Distinct":
            query = self.queryPart(part.p)
            return as_select(query).distinct()
        if part.name == "OrderBy":
            return self.queryOrderBy(part)
        if part.name == "Union":
            return self.queryUnion(part)
        if part.name == "Slice":
            return self.querySlice(part)
        if part.name == "LeftJoin":
            return self.queryLeftJoin(part)
        if part.name == "SelectQuery":
            return self.queryPart(part.p)

        e = f"Sparql part not implemented:{part}"
        raise SparqlNotImplementedError(e)

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
        query = self.queryPart(part)
        return self.exec(query)

    def getSQL(self, sparqlQuery, base:str|None=None, initNs:Mapping[str, Any] | None={}):
        from rdflib.plugins.sparql.parser import parseQuery
        from rdflib.plugins.sparql.algebra import translateQuery

        parsetree = parseQuery(sparqlQuery)
        queryobj = translateQuery(parsetree, base, initNs)
        with self.db.connect() as conn:
            query = self.queryPart(queryobj.algebra)
            return sql_pretty(query)

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

    ############################################## OLD STUFF #################################################