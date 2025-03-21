"""
The mapped SQL database as an RDF graph store object

.. note::
   TODO:
   
   * delay the UNION creation longer, so that we can use colforms to filter them

    * This could be complicated because you'd need to explode the BGPs 
      because each triple pattern becomes a set of selects, which you'd need to join



"""
from dataclasses import dataclass
from io import StringIO
from string import Formatter
from typing import Any, Generator, Mapping, Optional, List, Dict, Set, Tuple, TypeVar, Sequence, cast
import re

from rdflib import Variable
from rdflib.namespace import XSD, Namespace
from rdflib.plugins.sparql.parserutils import CompValue
import sqlalchemy
from sqlalchemy.sql import ColumnElement, func as sqlfunc, literal
import sqlalchemy.sql.operators
from sqlalchemy import ColumnElement, CompoundSelect, Label, literal, select, null, Dialect, Subquery
from sqlalchemy import union_all, or_ as sql_or, and_ as sql_and
from sqlalchemy import types as sqltypes, func as sqlfunc
from sqlalchemy.sql.expression import Select, Function, ClauseElement
from sqlalchemy.sql.selectable import _CompoundSelectKeyword, NamedFromClause, FromClause, Join
from sqlalchemy.sql.elements import NamedColumn
from rdflib_r2r.types import SQLQuery

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

class ImpossibleQueryException(Exception):
    "Raised when we get a query to convert that is legal, but we can't convert it to anything that could possibly return any results."
    pass

def sql_pretty(query:SQLQuery, dialect:Optional[Dialect] = None) -> str:
    import sqlparse

    qstr = str(query.compile(dialect=dialect, compile_kwargs={"literal_binds": True}))
    return sqlparse.format(qstr, reindent=True, keyword_case="upper")

def get_named_columns(query:Select|CompoundSelect|Subquery) -> Dict[str,ColumnElement]:
    return { col.name: col for col in query.exported_columns if isinstance(col, Label)}

def results_union(queries:Sequence[SQLQuery]) -> SQLQuery:

    if len(queries) == 1:
        return queries[0]
    elif len(queries) == 0:
        raise ImpossibleQueryException("Union of no queries")

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

@dataclass(eq=True, frozen=True, kw_only=True)
class TemplateInfo:
    is_iri: bool
    template: str
    columns: tuple[tuple[str,ColumnElement],...]
    table: NamedFromClause|None = None

    def column(self, name:str) -> ColumnElement:
        for n,c in self.columns:
            if n == name:
                return c
        raise KeyError(name)

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
    # Template expansion
    if column._annotations and "expansion_of" in column._annotations:
        exp = cast(TemplateInfo, column._annotations["expansion_of"])
        tab = exp.table
        if tab is not None:
            return tab
    
    # Label?
    if isinstance(column, Label):
        return get_column_table(column.element)

    # Direct table column
    if hasattr(column, "table") and column.table is not None:
        return column.table

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

def as_simple_select(query:SQLQuery) ->Select:
    if isinstance(query, Select) and is_simple_select(query):
        return query
    else:
        return wrap_in_select(query)

def wrap_in_select(query:SQLQuery) -> Select:
    # For all template-based columns,
    #   - Add a new exported column for each template column
    #   - Add an annotation on the template column pointing to the exported template columns.
    exp_info_specs = {}
    if isinstance(query, Select):
        names2cols = get_named_columns(query)
        extra_cols = []
        for c in query.exported_columns:
            if isinstance(c, NamedColumn):
                ei = get_template_expansion_info(c)
                if ei is not None:
                    col_vars = {}
                    for i, (cn,col) in enumerate(ei.columns):
                        name0 = f"{c.name}_K{i}"
                        name = name0
                        j = 0
                        while name in names2cols:
                            name = f"{name0}_{j}"
                            j += 1
                        extra_cols.append(col.label(name))
                        col_vars[cn] = name
                    exp_info_specs[c.name] = { "template": ei.template, "vars": col_vars, 'is_iri': ei.is_iri }

        query = query.with_only_columns(*query.exported_columns, *extra_cols)

    sq = query.subquery()

    result_cols = []
    for col in sq.exported_columns:
        ei_spec = exp_info_specs.get(col.key,None)
        if ei_spec is not None:
            col = col._annotate({
                "expansion_of": TemplateInfo(template= ei_spec["template"], is_iri=ei_spec["is_iri"], 
                    columns=tuple((cn, sq.exported_columns[v]) for cn,v in ei_spec["vars"].items() ))
            })
        col = col.label(col.key)
        result_cols.append(col)

    return select(*result_cols)

def get_template_expansion_info(e:Any) -> TemplateInfo|None:
    while isinstance(e,Label):
        e = e.element
    return e._annotations.get("expansion_of", None) if isinstance(e, ClauseElement) else None

def try_match_templates(a:ColumnElement, b:ColumnElement, eq:bool) -> ColumnElement|None:
    
    def try_template_to_literal_match(exp_info, lc):
        if hasattr(lc, "is_literal") and lc.is_literal:
            bv = lc.value
            if isinstance(bv, str):
                d = parse_with_template(bv, exp_info["template"])
                if d is not None:
                    eqs = []
                    for k,v in exp_info["columns"].items():
                        if k in d:
                            eqs.append(v == literal(d[k]) if eq else v != literal(d[k]))
                        else:
                            return None
                    return sql_and(*eqs) if eq else sql_or(*eqs)
        return None
    
    exp_info_a = get_template_expansion_info(a)
    exp_info_b = get_template_expansion_info(b)

    if exp_info_b is None:
        if exp_info_a is not None:
            return try_template_to_literal_match(exp_info_a, b)
        else:
            return None
    elif exp_info_a is None:
        return try_template_to_literal_match(exp_info_b, a)
        
    # Both have templates    
    if exp_info_a.template != exp_info_b.template:
        return None

    eqs = []
    for k, va in exp_info_a.columns:
        vb = exp_info_b.column(k)
        if vb is not None:
            eqs.append(va == vb if eq else va != vb)
        else:
            return None
    return sql_and(*eqs) if eq else sql_or(*eqs)


def equal(*expressions:ColumnElement, eq=True) -> Generator[ColumnElement[bool],None,None]:
    if expressions:
        e0, *es = expressions
        for e in es:
            eqty = try_match_templates(e0, e, eq)
            if eqty is not None:
                yield eqty
            else:    
                yield (e0 == e) if eq else (e0 != e) 

def collect_external_named_vars(part:CompValue, stop_at:CompValue|None, dest:Set[str]):
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


def already_includes(parent:FromClause, child:FromClause):
    if child is parent:
        return True
    if isinstance(parent,Join):
        if not parent.full:
            if already_includes(parent.left, child):
                return True
        if not parent.isouter:
            if already_includes(parent.right, child):
                return True
    return False

def combine_from_clauses(q:Select, exclude_from: FromClause|None = None):

    combined_from = None
    for f in q.get_final_froms():
        if exclude_from is not None and already_includes(exclude_from, f):
            continue
        if combined_from is None:
            combined_from = f
        else:
            combined_from = combined_from.join(f, onclause=literal(True))
    return combined_from

def merge_exported_columns(query1, query2) -> Tuple[List[NamedColumn], List[ColumnElement[bool]]]:
    allcols:List[NamedColumn] = []
    merge_conds:List[ColumnElement[bool]] = []
    names2cols1:dict[str,ColumnElement] = {}
    for c in query1.exported_columns:
        if isinstance(c, NamedColumn):
            names2cols1[c.name] = c
        allcols.append(c)

    for c in query2.exported_columns:
        if isinstance(c, NamedColumn):
            c1 = names2cols1.get(c.name,None)
            if c1 is not None:
                if not c.compare(c1):
                    merge_conds += list(equal(c1, c))
                continue
        allcols.append(c)
    return allcols, merge_conds

def expr_to_str(ex:ColumnElement):
    return str(ex.compile(compile_kwargs={"literal_binds": True}))

def format_template(template:str, tab:NamedFromClause, is_uri:bool) -> ColumnElement[str]:
    format_tuples = Formatter().parse(template)
    parts:List[ColumnElement] = []
    columns = []
    for prefix, colname, _, _ in format_tuples:
        if prefix != "":
            parts.append(literal(prefix))
        if colname:
            col = tab.c[colname]
            parts.append(col)
            columns.append((colname, col))

    return sqlfunc.concat(*parts)._annotate({
        "expansion_of": TemplateInfo(template=template, table=tab, columns= tuple(columns), is_iri=is_uri)
    })


def parse_with_template(s:str, template:str) -> Optional[Dict[str, str]]:
    format_tuples = Formatter().parse(template)
    pattern = StringIO()
    columns = []
    for prefix, colname, _, _ in format_tuples:
        if prefix:
            pattern.write(re.escape(prefix))
        if colname:
            columns.append(colname)
            pattern.write('(.*)')

    match = re.fullmatch(pattern.getvalue(), s)
    if not match:
        return None
    return {col: match[i+1] for i, col in enumerate(columns)}

    ############################################## OLD STUFF #################################################