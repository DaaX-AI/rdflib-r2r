from sqlalchemy.sql.elements import NamedColumn
from sqlalchemy.sql.selectable import NamedFromClause
from rdflib_r2r.r2r_mapping import toPython
from rdflib_r2r.conversion_utils import SQL_FUNC, SparqlNotImplementedError, XSDToSQL, already_includes, as_select, as_simple_select, collect_external_named_vars, combine_from_clauses, convert_pattern_to_like, equal, get_column_table, get_named_columns, merge_exported_columns, op, project_query, results_union, sql_pretty, wrap_in_select, ImpossibleQueryException
from rdflib_r2r.types import BGP, SQLQuery


import sqlalchemy
from rdflib import Literal, URIRef, Variable
from rdflib.plugins.sparql.parserutils import CompValue
from rdflib.term import Node
from rdflib.util import from_n3
from sqlalchemy import ColumnElement, Values, and_ as sql_and, column, except_, func as sqlfunc, literal, null, or_ as sql_or, select
from sqlalchemy.sql import ColumnElement, func as sqlfunc, literal
from sqlalchemy.sql.expression import case, distinct


from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Tuple, cast


class CurrentProject:
    project: CompValue
    named_tables: Dict[Tuple[str,str],NamedFromClause]
    vars_to_columns: Dict[str,ColumnElement]

    def __init__(self, project:CompValue):
        self.project = project
        self.named_tables = {}
        self.named_vars = set[str]();
        self.vars_to_columns = {}
        collect_external_named_vars(self.project.p, None, self.named_vars)

    def add_variables_to_columns(self, s:SQLQuery):
        for e in s.exported_columns:
            if isinstance(e, NamedColumn):
                k = e.name
                if k not in self.vars_to_columns:
                    self.vars_to_columns[k] = e


class QueryConversions(ABC):

    _current_project:CurrentProject|None = None

    @property
    def current_project(self) -> CurrentProject:
        assert self._current_project
        return self._current_project

    @abstractmethod
    def queryBGP(self, bgp: BGP) -> SQLQuery:
        ...


    def queryExpr(self, expr, var_cf:Mapping[str, ColumnElement]) -> ColumnElement:
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
                        return sqlfunc.cast(cf.expr, XSDToSQL[expr.iri])
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

            if (expr.name == "UnaryMinus"):
                cf = self.queryExpr(expr.expr, var_cf)
                return -cf

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
                return self.convertExists(expr)

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

    def convertExists(self, expr:CompValue):
        preexisting_keys = set(self.current_project.named_tables.keys())
        query = self.queryPart(expr.graph)
        self.current_project.named_tables = { k:v for k,v in self.current_project.named_tables.items() if k in preexisting_keys }
        external_vars = set[str]()
        assert self.current_project
        collect_external_named_vars(self.current_project.project.p, expr, external_vars)
        named_columns = get_named_columns(query)
        corr_froms = []
        wheres = []
        for n,c in named_columns.items():
            if n in external_vars:
                ec = self.current_project.vars_to_columns.get(n)
                if ec is None:
                    self.current_project.vars_to_columns[n] = c
                    f = get_column_table(c)
                else:
                    f = get_column_table(ec)
                    if not ec.compare(c):
                        wheres += list(equal(c, ec))
                if f is not None:
                    corr_froms.append(f)
        #corr_cols = [ c for n,c in named_columns.items() if n in external_vars ]
        #corr_froms = [ get_column_table(c) for c in corr_cols ]
        #corr_froms = [ f for f in corr_froms if f is not None ]

        sq = as_select(query).with_only_columns('*', maintain_column_froms=True).where(*wheres)
        return sq.exists().correlate(*corr_froms)

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
        #This doesn't always work; e.g. leads to multiple copies of the same table in from list...
        # if is_simple_select(part_query):
        #     return part_query
        # else:

        #Doing this in queryProject
        #return wrap_in_select(part_query)
        return part_query


    def queryJoin(self, part) -> SQLQuery:
        def is_empty(p):
            return p.name == "BGP" and not p.triples

        if is_empty(part.p1):
            return self.queryPart(part.p2)
        if is_empty(part.p2):
            return self.queryPart(part.p1)

        query1 = as_simple_select(self.queryPart(part.p1))
        query2 = as_simple_select(self.queryPart(part.p2))

        allcols, merge_conds = merge_exported_columns(query1, query2)

        froms1 = query1.get_final_froms()
        froms = list(froms1)
        for f in query2.get_final_froms():
            if not any(already_includes(f1,f) for f1 in froms1):
                froms.append(f)

        w2 = query2.whereclause
        if w2 is not None:
            merge_conds.append(w2)
        return query1.with_only_columns(*allcols).select_from(*froms).where(*merge_conds)

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

    def queryProject(self, part:CompValue, start = -1, limit = -1, distinct = False) -> SQLQuery:
        if part.name == "Slice":
            return self.queryProject(part.p, part.start, part.length, distinct)
        if part.name == "Distinct":
            return self.queryProject(part.p, start, limit, True)
        if part.name != "Project":
            raise ValueError(f"Expected Project, got {part.name}")

        old_project = self._current_project
        self._current_project = CurrentProject(part)
        try:
            part_query = self.queryPart(part.p)
            if distinct:
                part_query = as_select(part_query).distinct()
            if start >= 0:
                part_query = part_query.offset(start)
            if limit >= 0:
                part_query = part_query.limit(limit)

            expected_names = [str(v) for v in part.PV]
            result = project_query(part_query, expected_names)
            if old_project is not None:
                result = wrap_in_select(result)
            if old_project is not None:
                old_project.add_variables_to_columns(result)
            return result
        finally:
            self._current_project = old_project

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
        parts = []
        iqe = None
        for p in [ part.p1, part.p2 ]:
            try:
                parts.append(self.queryPart(p))
            except ImpossibleQueryException as e:
                iqe = e

        if parts:
            return results_union(parts)
        else:
            assert iqe
            raise iqe

    def queryLeftJoin(self, part) -> SQLQuery:

        query1 = as_simple_select(self.queryPart(part.p1))
        try:
            query2 = as_simple_select(self.queryPart(part.p2))
        except ImpossibleQueryException as e:
            return query1

        allcols, merge_conds = merge_exported_columns(query1, query2)
        named_cols = { c.name: c for c in allcols }
        if part.expr is not None:
            if part.expr.name != "TrueFilter":
                merge_conds.append(self.queryExpr(part.expr, named_cols))

        from1 = combine_from_clauses(query1)
        from2 = combine_from_clauses(query2, exclude_from=from1)

        assert from1 is not None
        assert from2 is not None

        wc2 = query2.whereclause
        if wc2 is not None:
            merge_conds.append(wc2)
        onclause = sql_and(*merge_conds) if merge_conds else literal(True)
        joined_from = from1.join(from2, isouter=True, onclause=onclause)
        return query1.with_only_columns(*allcols).select_from(joined_from)
    
    def queryValues(self, part) -> SQLQuery:
        vals: list[dict[Variable, Node]] = part.res
        vars = [ v for v in part.res[0].keys() ]

        value_rows = []
        for val_dict in vals:
            value_rows.append(tuple(self.queryExpr(val_dict[var], {}) for var in vars))

        values_clause = Values(*[column(str(v)) for v in vars]).data(value_rows).alias('vals')
        result = select(*[col.label(col.name) for col in values_clause.exported_columns])
        return result

    def queryPart(self, part:CompValue) -> SQLQuery:
        if part.name == "BGP":
            return self.queryBGP(part.triples)
        if part.name == "Filter":
            return self.queryFilter(part)
        if part.name == "Extend":
            return self.queryExtend(part)
        if part.name == "Project" or part.name == "Slice" or part.name == "Distinct":
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
        if part.name == "OrderBy":
            return self.queryOrderBy(part)
        if part.name == "Union":
            return self.queryUnion(part)
        if part.name == "LeftJoin":
            return self.queryLeftJoin(part)
        if part.name == "values":
            return self.queryValues(part)
        if part.name == "SelectQuery":
            return self.queryPart(part.p)

        e = f"Sparql part not implemented:{part}"
        raise SparqlNotImplementedError(e)


