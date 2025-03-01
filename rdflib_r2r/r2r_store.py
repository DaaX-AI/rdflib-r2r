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
from io import StringIO
from typing import Any, Generator, Mapping, Optional, List, Dict, Set, cast, TypeVar, Sequence
import logging
import base64
import re

from rdflib import Graph, URIRef, Literal, BNode, Variable
from rdflib.namespace import XSD, Namespace
from rdflib.store import Store
from rdflib.util import from_n3
from rdflib.term import Node
from rdflib.plugins.sparql.parserutils import CompValue
import sqlalchemy
import sqlalchemy.sql.operators
from sqlalchemy import ColumnElement, CompoundSelect, Label,  except_, literal, select, null, literal_column, Dialect, Subquery
from sqlalchemy import union_all, or_ as sql_or, and_ as sql_and
from sqlalchemy import types as sqltypes, func as sqlfunc
from sqlalchemy.sql.expression import Select, distinct, case, Function, ClauseElement
from sqlalchemy.sql.selectable import _CompoundSelectKeyword
from sqlalchemy.sql.elements import NamedColumn
from sqlalchemy.engine import Engine

from rdflib_r2r.expr_template import ExpressionTemplate, SubForm
from rdflib_r2r.types import SQLQuery, BGP
from rdflib_r2r.r2r_mapping import iri_safe, toPython

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

SQL_FUNC = Namespace("http://daax.ai/sqlfunc/")

def sql_pretty(query:SQLQuery, dialect:Optional[Dialect] = None) -> str:
    import sqlparse

    qstr = str(query.compile(dialect=dialect, compile_kwargs={"literal_binds": True}))
    return sqlparse.format(qstr, reindent=True, keyword_case="upper")

def get_named_columns(query:Select|CompoundSelect|Subquery) -> Dict[str,ColumnElement]:
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

def get_column_table(column: ColumnElement):
    """
    Given a SQLAlchemy ColumnElement, return the associated Table if it is a column access.
    
    This function handles:
    - Direct table columns (`table.c.column_name`)
    - Aliased table columns (`aliased(table).c.column_name`)
    - ORM-mapped columns
    - Columns wrapped in expressions
    
    Returns:
        Table: The SQLAlchemy Table object if found, otherwise None.
    """
    # Direct table column
    if hasattr(column, "table") and column.table is not None:
        return column.table

    # Aliased columns
    if hasattr(column, "original") and column.original is not None:
        return get_column_table(column.original)

    return None

def is_simple_select(stmt:SQLQuery) -> bool:
    if not isinstance(stmt, Select):
        stmt = as_select(stmt)
    
    aggregation_funcs = {"sum", "avg", "count", "min", "max", "group_concat_node"}

    # Check if an expression involves any aggregation
    def has_aggregation_expression(expr):
        if isinstance(expr, Function) and expr.name in aggregation_funcs:
            return True

        if isinstance(expr, ClauseElement):
            # Check if any sub-expression or nested function has aggregation
            for sub_expr in expr.get_children():
                if has_aggregation_expression(sub_expr):
                    return True
        return False

    # Check if the query has aggregation
    if any(has_aggregation_expression(col) for col in stmt.selected_columns):
        return False

    # Check for GROUP BY clauses
    if len(stmt._group_by_clauses) > 0:
        return False

    # Check for LIMIT/OFFSET
    if stmt._limit is not None or stmt._offset is not None:
        return False
    
    if stmt._fetch_clause is not None:
        return False
    
    if len(stmt._having_criteria) > 0:
        return False
    
    if stmt._distinct:
        return False

    return True


def as_select(query:SQLQuery) ->Select:
    return query if isinstance(query, Select) else wrap_in_select(query)

def wrap_in_select(query:SQLQuery) -> Select:
    sq = query.subquery()
    return select(*[col.label(col.key) for col in sq.exported_columns])


def equal(*expressions, eq=True) -> Generator[ColumnElement,None,None]:
    if expressions:
        e0, *es = expressions
        for e in es:
            yield (e0 == e) if eq else (e0 != e) #TODO: disassemble templates

def collect_external_named_vars(part:CompValue, stop_at:CompValue, dest:Set[str]):
    if part is stop_at:
        return
    
    if isinstance(part, CompValue):
        if part.name == "BGP":
            for t in part.triples:
                for v in t:
                    if isinstance(v, Variable):
                        dest.add(str(v))

        elif part.name == "Project":
            for v in part.PV:
                dest.add(str(v))

        elif part.name == "Extend":
            dest.add(str(part.var))
            collect_external_named_vars(part.p, stop_at, dest)

        elif hasattr(part, "p"):
            collect_external_named_vars(part.p, stop_at, dest)

        if hasattr(part, "p1"):
            collect_external_named_vars(part.p1, stop_at, dest)
            collect_external_named_vars(part.p2, stop_at, dest)


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
    
def print_algebra(sparql:str, initNs:Mapping[str, Any] ={}):
    from rdflib.plugins.sparql.parser import parseQuery
    from rdflib.plugins.sparql.algebra import translateQuery, pprintAlgebra

    parsetree = parseQuery(sparql)
    queryobj = translateQuery(parsetree, None, initNs)
    pprintAlgebra(queryobj)

Element = TypeVar("Element")

def iter_opt(o:Optional[Element]) -> Generator[Element,None,None]:
    if o is not None:
        yield o

def convert_pattern_to_like(regex:str) -> str:
    like = StringIO()
    i = 0
    if not regex.endswith('$') or not regex.startswith('^'):
        raise ValueError("Only anchored regexes can be converted to LIKE patterns")
    regex = regex[1:-1]
    while i < len(regex):
        c = regex[i]
        if c == '.':
            if i+1 < len(regex) and regex[i+1] == '*':
                like.write('%')
                i += 1
            else:
                like.write('_')
        elif c == '\\':
            if regex[i+1].isdigit():
                raise ValueError("Groups not supported in LIKE patterns")
            like.write('[')
            like.write(regex[i+1])
            like.write(']')
            i += 1
        elif c in { '^', '$', '*', '+', '?', '{', '}', '|', '(', ')'}:
            raise ValueError(f"Unsupported regex character: {c}")
        elif c == '[':
            like.write('[')

            i += 1
            while i < len(regex) and regex[i] != ']':
                like.write(regex[i])
                i += 1
            if regex[i] == ']':
                like.write(']')
            else:
                raise ValueError("Unmatched '[' in regex")
        else:
            like.write(c)

        i += 1

    return like.getvalue()

class R2RStore(Store, ABC):
    """
    Args:
        db: SQLAlchemy engine.
    """

    current_project:CompValue|None = None

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
        if hasattr(expr, "name"):

            if (expr.name in agg_funcs):
                if expr.name == "Aggregate_Count" and expr.vars == '*':
                    return sqlfunc.count()
                sub = self.queryExpr(expr.vars, var_cf)
                if expr.name == "Aggregate_Sample":
                    return sub
                func = agg_funcs[expr.name]
                # if (len(sub.cols) == 1) and (expr.name == "Aggregate_Count"):
                #     # Count queries don't need full node expression
                #     return func(sub.cols[0])
                # else:
                if expr.distinct == 'DISTINCT':
                    sub = distinct(sub)
                return func(sub)

            if (expr.name == "RelationalExpression"):
                if expr.op == "IN":
                    a = self.queryExpr(expr.expr, var_cf)
                    b = [ self.queryExpr(elt, var_cf) for elt in expr.other ]
                    return a.in_(b)
                a = self.queryExpr(expr.expr, var_cf)
                b = self.queryExpr(expr.other, var_cf)
                return op(expr.op, a, b)

            math_expr_names = ["MultiplicativeExpression", "AdditiveExpression"]
            if (expr.name in math_expr_names):
                # TODO: ternary ops?
                a = self.queryExpr(expr.expr, var_cf)
                for op_sym, other in zip(expr.op, expr.other):
                    b = self.queryExpr(other, var_cf)
                    a = op(op_sym, a, b)
                return a

            if (expr.name == "ConditionalAndExpression"):
                exprs = [self.queryExpr(e, var_cf) for e in [expr.expr] + expr.other]
                return sql_and(*exprs)

            if (expr.name == "ConditionalOrExpression"):
                exprs = [self.queryExpr(e, var_cf) for e in [expr.expr] + expr.other]
                return sql_or(*exprs)

            if (expr.name == "Function"):
                # TODO: it would be super cool to do UDFs here
                if expr.iri in XSDToSQL:
                    for e in expr.expr:
                        cf = self.queryExpr(e, var_cf)
                        return sqlfunc.cast(cf.expr(), XSDToSQL[expr.iri])
                if expr.iri.startswith(SQL_FUNC):
                    func_name = expr.iri[len(SQL_FUNC):]
                    func = getattr(sqlfunc, func_name, None)
                    if func is None:
                        raise SparqlNotImplementedError(f"SQL function not implemented: {expr.iri}")
                    return func(*[self.queryExpr(e, var_cf) for e in expr.expr])

            if (expr.name == "UnaryNot"):
                arg = expr.expr
                if isinstance(arg, CompValue):
                    if arg.name == "UnaryNot":
                        return self.queryExpr(arg.expr, var_cf)
                    elif arg.name == "Builtin_BOUND":
                        return self.queryExpr(arg.arg, var_cf).is_(None)
                
                cf = self.queryExpr(expr.expr, var_cf)
                return sqlalchemy.not_(cf) 
            
            if (expr.name == "Builtin_BOUND"):
                cf = self.queryExpr(expr.arg, var_cf)
                return sqlfunc.not_(cf.is_(None))
            
            if (expr.name == "Builtin_REGEX"):
                if set(expr.flags) != {"i","s"}:
                    raise SparqlNotImplementedError("We only support case-insensitive, dotall regexes")
                if not isinstance(expr.pattern, Literal):
                    raise SparqlNotImplementedError("Non-literal regex pattern not supported")
                try:
                    like_pattern = convert_pattern_to_like(expr.pattern.toPython())
                except ValueError as e:
                    raise SparqlNotImplementedError(f"Cannot convert regex pattern to like pattern `{expr.pattern}`: {str(e)}") from e
                
                cf = self.queryExpr(expr.text, var_cf)
                return cf.like(like_pattern)
            if (expr.name == "Builtin_IF"):
                cases = []
                if_expr = expr
                while hasattr(if_expr, "name") and (if_expr.name == "Builtin_IF"):
                    cases.append((self.queryExpr(if_expr.arg1, var_cf), self.queryExpr(if_expr.arg2, var_cf)))
                    if_expr = if_expr.arg3
                return case(*cases, else_=self.queryExpr(if_expr, var_cf))
            
            if expr.name == "Builtin_EXISTS":
                query = self.queryPart(expr.graph)
                external_vars = set[str]()
                assert self.current_project
                collect_external_named_vars(self.current_project.p, expr, external_vars)
                named_columns = get_named_columns(query)
                corr_cols = [ c for n,c in named_columns.items() if n in external_vars ]
                corr_froms = [ get_column_table(c) for c in corr_cols ]
                sq = as_select(query).with_only_columns('*', maintain_column_froms=True).correlate(*corr_froms)
                return sq.exists()
            
            if expr.name == "Builtin_NOTEXISTS":
                ex = self.queryExpr(CompValue("Builtin_EXISTS", graph=expr.graph), var_cf)
                return sqlalchemy.not_(ex)
            
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

        clause = self.queryExpr(part.expr, named_cols)

        # Filter should be HAVING for aggregates
        def is_aggregate(p):
            if p.name == "ToMultiSet":
                return False # Subquery
            if p.name == "AggregateJoin":
                return True
            if "p" not in p:
                return False
            return is_aggregate(p.p)
        
        if is_aggregate(part.p):
            return part_query.having(clause)
        else:
            return part_query.where(clause)
        
    def queryToMultiset(self, part) -> SQLQuery:   
        part_query = self.queryPart(part.p)
        if is_simple_select(part_query):
            return part_query
        else:
            return wrap_in_select(part_query)

    def queryJoin(self, part) -> SQLQuery:
        def is_empty(p):
            return p.name == "BGP" and not p.triples
        
        if is_empty(part.p1):
            return self.queryPart(part.p2)
        if is_empty(part.p2):
            return self.queryPart(part.p1)
        
        query1 = self.queryPart(part.p1)
        query2 = self.queryPart(part.p2)

        j1 = query1.alias('j1')
        j2 = query2.alias('j2')

        allcols = []
        names1 = set()
        for c in j1.exported_columns:
            if isinstance(c, NamedColumn):
                names1.add(c.name)
                c = c.label(c.name)
            allcols.append(c)

        common_names = set()
        for c in j2.exported_columns:
            if isinstance(c, NamedColumn):
                if c.name in names1:
                    common_names.add(c.name)
                    continue
                c = c.label(c.name)
            allcols.append(c)

        wheres = [ j1.exported_columns[n] == j2.exported_columns[n] for n in common_names ]
        return select(*allcols).where(*wheres)

    def queryAggregateJoin(self, agg) -> SQLQuery:
        # Assume agg.p is always a Group
        group_expr, group_part = agg.p.expr, agg.p.p
        part_query = self.queryPart(group_part)
        named_cols = get_named_columns(part_query)

        # Get aggregate column expressions
        aggs = [ self.queryExpr(a, named_cols).label(str(a.res)) for a in agg.A ]
        groups = [ 
            self.queryExpr(ge, named_cols) for ge in (group_expr or [])
        ]

        return as_select(part_query).with_only_columns(*aggs, maintain_column_froms=True).group_by(*groups)

    def queryExtend(self, part) -> SQLQuery:
        part_query = self.queryPart(part.p)
        part_query = as_select(part_query)

        cf = self.queryExpr(part.expr, get_named_columns(part_query))
        return part_query.with_only_columns(*part_query.exported_columns, cf.label(str(part.var)))

    def queryProject(self, part:CompValue) -> SQLQuery:
        old_project = self.current_project
        self.current_project = part
        try:
            part_query = self.queryPart(part.p)
            expected_names = [str(v) for v in part.PV]
            return project_query(part_query, expected_names)
        finally:
            self.current_project = old_project
    
    def queryOrderBy(self, part) -> SQLQuery:
        part_query = self.queryPart(part.p)
        ncs = get_named_columns(part_query)

        ordering:List[ColumnElement] = []
        for oc in part.expr:
            expr = self.queryExpr(oc.expr, ncs)
            if oc.order == "DESC":
                expr = sqlalchemy.desc(expr)
            ordering.append(expr)

        return part_query.order_by(*ordering)

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

        #XXX Horribly broken!
        query1 = self.queryPart(part.p1)
        query2 = self.queryPart(part.p2)

        j1 = query1.alias('j1')
        j2 = query2.alias('j2')

        allcols = []
        names1 = set()
        for c in j1.exported_columns:
            if isinstance(c, NamedColumn):
                names1.add(c.name)
                c = c.label(c.name)
            allcols.append(c)

        common_names = set()
        for c in j2.exported_columns:
            if isinstance(c, NamedColumn):
                if c.name in names1:
                    common_names.add(c.name)
                    continue
                c = c.label(c.name)
            allcols.append(c)


        ons = [ j1.exported_columns[n] == j2.exported_columns[n] for n in common_names ]
        return  as_select(query1).outerjoin(as_select(query2).subquery(), onclause=sql_and(*ons))

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
            return self.queryToMultiset(part)
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
        sql_query = self.queryPart(queryobj.algebra)
        return sql_pretty(sql_query, dialect=self.db.dialect)

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