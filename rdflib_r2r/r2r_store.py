"""
The mapped SQL database as an RDF graph store object

.. note::
   TODO:
   
   * delay the UNION creation longer, so that we can use colforms to filter them

    * This could be complicated because you'd need to explode the BGPs 
      because each triple pattern becomes a set of selects, which you'd need to join



"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from os import linesep
from typing import Generic, Iterator, Optional, Iterable, List, Tuple, Union, Dict, NamedTuple, cast, TypeVar
import logging
import base64
from decimal import Decimal
import re
import math

from rdflib import Graph, URIRef, Literal, BNode, Variable
from rdflib.namespace import XSD, Namespace
from rdflib.store import Store
from rdflib.util import from_n3
from rdflib.term import Node
from rdflib.plugins.sparql.parserutils import CompValue
import sqlalchemy
import sqlalchemy.sql.operators
from sqlalchemy import MetaData, select, text, null, literal_column, literal, TableClause, Subquery, Table, ClauseElement, Function
from sqlalchemy import union_all, or_ as sql_or, and_ as sql_and
from sqlalchemy import schema as sqlschema, types as sqltypes, func as sqlfunc
from sqlalchemy.sql.expression import GenerativeSelect, Select, ColumnClause
from sqlalchemy.sql.selectable import ScalarSelect, CompoundSelect, NamedFromClause

from sqlalchemy.engine import Engine, Connection

from rdflib_r2r.expr_template import ExpressionTemplate
from rdflib_r2r.types import Triple, BGP
from rdflib_r2r.r2r_mapping import R2RMapping, iri_safe, toPython

class SubForm(NamedTuple):
    """A template for generating an RDF node from an _external_ list of SQL expressions"""
    #: Offsets of variables in a SQL Select statement to use in RDF node template
    select_var_indices: Iterable[int]
    #: RDF node template; booleans indicate whether to SQL-escape the columns
    form: Tuple[Union[str, bool, None], ...]

class SelectSubForm(NamedTuple):
    select: Select #: 
    subforms: List[SubForm] #: 

class GenerativeSelectSubForm(NamedTuple):
    select: GenerativeSelect #: 
    subforms: List[SubForm] #: 

SelectType = TypeVar("SelectType", bound=GenerativeSelect)

@dataclass
class SelectVarSubForm(Generic[SelectType]):
    select: SelectType #: 
    #: A map of RDF variables to the subforms that generate them from SQL expressions
    varsubforms: Dict[Variable, SubForm]

    def as_tuple(self) -> tuple[SelectType, Dict[Variable, SubForm]]:
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


def sql_safe(literal):
    # return "<ENCODE>" + sqlfunc.cast(literal, sqltypes.VARCHAR) + "</ENCODE>"
    # This is a terrible idea due to the large number of chars.
    # ... but the alternative messes up SPARQL query rewriting

    # TEMP: DISABLE FOR DEBUGGING
    literal = sqlfunc.replace(literal, " ", "%20")
    literal = sqlfunc.replace(literal, "/", "%2F")
    literal = sqlfunc.replace(literal, "(", "%28")
    literal = sqlfunc.replace(literal, ")", "%29")
    literal = sqlfunc.replace(literal, ",", "%2C")
    literal = sqlfunc.replace(literal, ":", "%3A")

    return literal


def sql_pretty(query):
    import sqlparse

    qstr = str(query.compile(compile_kwargs={"literal_binds": True}))
    return sqlparse.format(qstr, reindent=True, keyword_case="upper")


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

    @staticmethod
    def _get_col(dbtable:Table|NamedFromClause, colname:str, template=False) -> ColumnClause|Function:
        if type(dbtable) is sqlschema.Table:
            dbcol = dbtable.c[colname.strip('"')]

            if isinstance(dbcol.type, sqltypes.Numeric):
                if dbcol.type.precision:
                    if template:
                        # Binary data
                        return sqlfunc.hex(dbcol)
                    else:
                        return literal_column(f'"{dbtable.name}".{colname}')

            if (not template) and isinstance(dbcol.type, sqltypes.CHAR):
                if dbcol.type.length:
                    l = dbcol.type.length
                    return sqlfunc.substr(dbcol + " " * l, 1, l)

            return dbcol
        else:
            return literal_column(f'"{dbtable.name}".{colname}')


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
    def queryBGP(self, conn: Connection, bgp: BGP) -> SelectVarSubForm:
        ...

 
    def queryExpr(self, conn: Connection, expr, var_cf:dict[Variable, ExpressionTemplate]) -> ExpressionTemplate:
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
            sub = self.queryExpr(conn, expr.vars, var_cf)
            if expr.name == "Aggregate_Sample":
                return sub
            func = agg_funcs[expr.name]
            if (len(sub.cols) == 1) and (expr.name == "Aggregate_Count"):
                # Count queries don't need full node expression
                return ExpressionTemplate.from_expr(func(sub.cols[0]))
            else:
                return ExpressionTemplate.from_expr(func(sub.expr()))

        if hasattr(expr, "name") and (expr.name == "RelationalExpression"):
            a = self.queryExpr(conn, expr.expr, var_cf)
            b = self.queryExpr(conn, expr.other, var_cf)
            return ExpressionTemplate.op(expr.op, a, b)

        math_expr_names = ["MultiplicativeExpression", "AdditiveExpression"]
        if hasattr(expr, "name") and (expr.name in math_expr_names):
            # TODO: ternary ops?
            a = self.queryExpr(conn, expr.expr, var_cf)
            for other in expr.other:
                b = self.queryExpr(conn, other, var_cf)
                return ExpressionTemplate.op(expr.op[0], a, b)

        if hasattr(expr, "name") and (expr.name == "ConditionalAndExpression"):
            exprs = [self.queryExpr(conn, e, var_cf) for e in [expr.expr] + expr.other]
            return ExpressionTemplate.from_expr(sql_and(*[e.expr() for e in exprs]))

        if hasattr(expr, "name") and (expr.name == "ConditionalOrExpression"):
            exprs = [self.queryExpr(conn, e, var_cf) for e in [expr.expr] + expr.other]
            return ExpressionTemplate.from_expr(sql_or(*[e.expr() for e in exprs]))

        if hasattr(expr, "name") and (expr.name == "Function"):
            # TODO: it would be super cool to do UDFs here
            if expr.iri in XSDToSQL:
                for e in expr.expr:
                    cf = self.queryExpr(conn, e, var_cf)
                    val = sqlfunc.cast(cf.expr(), XSDToSQL[expr.iri])
                    return ExpressionTemplate.from_expr(val)

        if hasattr(expr, "name") and (expr.name == "UnaryNot"):
            cf = self.queryExpr(conn, expr.expr, var_cf)
            return ExpressionTemplate.from_expr( sqlalchemy.not_(cf.expr()) ) 
        
        if hasattr(expr, "name") and (expr.name == "Builtin_BOUND"):
            cf = self.queryExpr(conn, expr.arg, var_cf)
            return ExpressionTemplate.from_expr( cf.expr().is_(None) ) 

        if isinstance(expr, str) and (expr in var_cf):
            return var_cf[expr]
        if isinstance(expr, URIRef):
            return ExpressionTemplate.from_expr(expr.n3())
        if isinstance(expr, Literal):
            return ExpressionTemplate.from_expr(expr.toPython())
        if isinstance(expr, str):
            return ExpressionTemplate.from_expr(toPython(cast(Node,from_n3(expr))))

        e = f'Expr not implemented: {getattr(expr, "name", None).__repr__()} {expr}'
        raise SparqlNotImplementedError(e)

    def queryFilter(self, conn: Connection, part:CompValue) -> SelectVarSubForm:
        part_query, var_subform = self.queryPart(conn, part.p).as_tuple()

        if getattr(part.expr, "name", None) == "Builtin_NOTEXISTS":
            # This is weird, but I guess that's how it is
            query2, var_subform2 = self.queryPart(conn, part.expr.graph).as_tuple()

            var_colforms = {}
            cols1 = list(part_query.exported_columns)
            for v, sf1 in var_subform.items():
                var_colforms.setdefault(v, []).append(ExpressionTemplate.from_subform(cols1, *sf1))
            cols2 = list(query2.exported_columns)
            for v, sf2 in var_subform2.items():
                var_colforms.setdefault(v, []).append(ExpressionTemplate.from_subform(cols2, *sf2))

            where = [eq for cs in var_colforms.values() for eq in ExpressionTemplate.equal(*cs)]
            return SelectVarSubForm(part_query.filter(~query2.where(*where).exists()), var_subform)

        cols = list(getattr(part_query, "exported_columns", part_query.c))
        var_cf = {v: ExpressionTemplate.from_subform(cols, *sf) for v, sf in var_subform.items()}
        logging.warn(('Building filter clause from', part.expr, var_cf))
        clause = self.queryExpr(conn, part.expr, var_cf).expr()
        logging.warn(('Built filter clause', str(clause.compile())))

        # Filter should be HAVING for aggregates
        if part.p.name == "AggregateJoin":
            return SelectVarSubForm(part_query.having(clause), var_subform)
        else:
            return SelectVarSubForm(part_query.where(clause), var_subform)

    def queryJoin(self, conn: Connection, part) -> SelectVarSubForm:
        query1, var_subform1 = self.queryPart(conn, part.p1).as_tuple()
        query2, var_subform2 = self.queryPart(conn, part.p2).as_tuple()
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

    def queryAggregateJoin(self, conn: Connection, agg) -> SelectVarSubForm:
        # Assume agg.p is always a Group
        group_expr, group_part = agg.p.expr, agg.p.p
        part_query, var_subform = self.queryPart(conn, group_part).as_tuple()
        cols = part_query.c
        var_cf = {v: ExpressionTemplate.from_subform(cols, *sf) for v, sf in var_subform.items()}

        # Get aggregate column expressions
        var_agg = {a.res: self.queryExpr(conn, a, var_cf) for a in agg.A}
        groups = [
            c
            for e in (group_expr or [])
            for c in self.queryExpr(conn, e, var_cf).cols
            if type(c) != str
        ]

        subforms, allcols = ExpressionTemplate.to_subforms_columns(*var_agg.values())
        query = select(*allcols).group_by(*groups)
        return SelectVarSubForm(query, dict(zip(var_agg, subforms)))

    def queryExtend(self, conn: Connection, part) -> SelectVarSubForm:
        part_query, var_subform = self.queryPart(conn, part.p).as_tuple()
        assert isinstance(part_query, Select)  # ?
        cols = list(part_query.exported_columns)
        var_cf = {v: ExpressionTemplate.from_subform(cols, *sf) for v, sf in var_subform.items()}

        cf = self.queryExpr(conn, part.expr, var_cf)
        idxs = []
        for c in cf.cols:
            if c in cols:
                idxs.append(cols.index(c))
            else:
                idxs.append(len(cols))
                cols.append(c)

        var_subform[part.var] = SubForm(idxs, cf.form)

        return SelectVarSubForm(part_query.with_only_columns(*(cols + list(cf.cols))), var_subform)

    def queryProject(self, conn: Connection, part) -> SelectVarSubForm:
        part_query, var_subform = self.queryPart(conn, part.p).as_tuple()
        var_subform = {v: sf for v, sf in var_subform.items() if v in part.PV}
        cols = list(part_query.exported_columns)
        colforms = [ExpressionTemplate.from_subform(cols, *sf) for sf in var_subform.values()]
        subforms, allcols = ExpressionTemplate.to_subforms_columns(*colforms)
        part_query = part_query.with_only_columns(*allcols)
        return SelectVarSubForm(part_query, dict(zip(var_subform, subforms)))

    def queryOrderBy(self, conn: Connection, part) -> SelectVarSubForm:
        part_query, var_subform = self.queryPart(conn, part.p).as_tuple()
        cols = list(part_query.exported_columns)
        var_cf = {v: ExpressionTemplate.from_subform(cols, *sf) for v, sf in var_subform.items()}

        ordering = []
        for e in part.expr:
            expr_cf = self.queryExpr(conn, e.expr, var_cf)
            if expr_cf.form[0] != '<':
                for col in expr_cf.cols:
                    if e.order == "DESC":
                        col = sqlalchemy.desc(col)
                    ordering.append(col)
            else:
                ordering.append(expr_cf.expr())

        return SelectVarSubForm(part_query.order_by(*ordering), var_subform)

    def queryUnion(self, conn: Connection, part) -> SelectVarSubForm:
        query1, var_subform1 = self.queryPart(conn, part.p1).as_tuple()
        query2, var_subform2 = self.queryPart(conn, part.p2).as_tuple()

        all_vars = set(var_subform1) | set(var_subform2)

        cols1, cols2 = list(query1.c), list(query2.c)
        allcols1, allcols2 = [], []
        var_sf = {}
        for i, v in enumerate(all_vars):
            # TODO: if forms are identical, don't convert to expression
            var_sf[v] = [i], [None]
            if v in var_subform1:
                e1 = ExpressionTemplate.from_subform(cols1, *var_subform1[v]).expr()
            else:
                e1 = null()
            allcols1.append(e1.label(str(v)))
            if v in var_subform2:
                e2 = ExpressionTemplate.from_subform(cols2, *var_subform2[v]).expr()
            else:
                e2 = null()
            allcols2.append(e2.label(str(v)))
        query1 = select(*allcols1)
        query2 = select(*allcols2)

        return SelectVarSubForm(union_all(query1, query2), var_sf)

    def querySlice(self, conn: Connection, part) -> SelectVarSubForm:
        query, var_subform = self.queryPart(conn, part.p).as_tuple()
        if part.start:
            query = query.offset(part.start)
        if part.length:
            query = query.limit(part.length)
        return SelectVarSubForm(query, var_subform)

    def queryLeftJoin(self, conn: Connection, part) -> SelectVarSubForm:

        query1, var_subform1 = self.queryPart(conn, part.p1).as_tuple()
        query2, var_subform2 = self.queryPart(conn, part.p2).as_tuple()
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

    def queryPart(self, conn: Connection, part:CompValue) -> SelectVarSubForm:
        if part.name == "BGP":
            return self.queryBGP(conn, part.triples)
        if part.name == "Filter":
            return self.queryFilter(conn, part)
        if part.name == "Extend":
            return self.queryExtend(conn, part)
        if part.name == "Project":
            return self.queryProject(conn, part)
        if part.name == "Join":
            return self.queryJoin(conn, part)
        if part.name == "AggregateJoin":
            return self.queryAggregateJoin(conn, part)
        if part.name == "ToMultiSet":
            # no idea what this should do
            return self.queryPart(conn, part.p)
        if part.name == "Minus":
            q1, v1 = self.queryPart(conn, part.p1).as_tuple()
            q2, _ = self.queryPart(conn, part.p2).as_tuple()
            return SelectVarSubForm(q1.except_(q2), v1)
        if part.name == "Distinct":
            query, var_subform = self.queryPart(conn, part.p).as_tuple()
            return SelectVarSubForm(query.distinct(), var_subform)
        if part.name == "OrderBy":
            return self.queryOrderBy(conn, part)
        if part.name == "Union":
            return self.queryUnion(conn, part)
        if part.name == "Slice":
            return self.querySlice(conn, part)
        if part.name == "LeftJoin":
            return self.queryLeftJoin(conn, part)
        if part.name == "SelectQuery":
            return self.queryPart(conn, part.p)

        e = f"Sparql part not implemented:{part}"
        raise SparqlNotImplementedError(e)

    def exec(self, query):
        with self.db.connect() as conn:
            logging.warn("Executing:\n" + sql_pretty(query))
            # raise Exception
            results = conn.execute(query)
            rows = list(results)
            keys = [Variable(v) for v in results.keys()]
            logging.warn(f"Got {len(rows)} rows of {keys}")
            first = True
            for vals in rows:
                if first:
                    logging.warn(f"First row: {vals}")
                    first = False
                yield dict(zip(keys, [self.make_node(v) for v in vals]))

    @staticmethod
    def _apply_subforms(query, var_subform):
        if isinstance(query, Select):
            cols = [
                ExpressionTemplate.from_subform(query.exported_columns, *sf).expr().label(str(var))
                for var, sf in var_subform.items()
            ]
            return query.with_only_columns(*cols)
        else:
            cols = [
                ExpressionTemplate.from_subform(query.c, *sf).expr().label(str(var))
                for var, sf in var_subform.items()
            ]
            return select(*cols)

    def evalPart(self, part):
        with self.db.connect() as conn:
            query, var_subform = self.queryPart(conn, part).as_tuple()
            query = self._apply_subforms(query, var_subform)
        return self.exec(query)

    def getSQL(self, sparqlQuery, base=None, initNs={}):
        from rdflib.plugins.sparql.parser import parseQuery
        from rdflib.plugins.sparql.algebra import translateQuery

        parsetree = parseQuery(sparqlQuery)
        queryobj = translateQuery(parsetree, base, initNs)
        with self.db.connect() as conn:
            query, var_subform = self.queryPart(conn, queryobj.algebra).as_tuple()
            sqlquery = self._apply_subforms(query, var_subform)
            return sql_pretty(sqlquery)

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