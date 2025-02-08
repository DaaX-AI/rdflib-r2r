from ast import Expr
from dataclasses import dataclass, replace
from io import StringIO
import logging
import re
from string import Formatter
from typing import Any, Dict, Generator, List, Literal as LiteralType, Optional, Type, cast
from rdflib import RDF, URIRef, Variable
from rdflib.term import Node
from rdflib.paths import AlternativePath, SequencePath, InvPath
from sqlalchemy import types as sqltypes, MetaData
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.sql import Select, ColumnElement, select, literal_column, literal, func as sqlfunc
from sqlalchemy.sql.selectable import NamedFromClause, FromClauseAlias, FromClause
from rdflib_r2r.expr_template import ExpressionTemplate, SubForm
from rdflib_r2r.r2r_mapping import R2RMapping, _get_table, rr, toPython
from rdflib_r2r.r2r_store import R2RStore, SelectVarSubForm, sql_and
from rdflib_r2r.types import BGP, SearchQuery
import queue

@dataclass(frozen=True, eq=True, kw_only=True)
class Row:
    subject: Node
    table: NamedFromClause

@dataclass(frozen=True, eq=True, kw_only=True)
class VariableExpression:
    expression: ColumnElement
    template: str|None

@dataclass
class ProcessingState:
    rows: Dict[Node,Row]
    var_expressions: Dict[Variable, ColumnElement]
    wheres: List[ColumnElement[bool]]
    triples: List[SearchQuery]

def get_col(tab:NamedFromClause, col_name:str) -> ColumnElement:
    return tab.c[col_name]

def format_template(template:str, tab:NamedFromClause) -> ColumnElement[str]:
    format_tuples = Formatter().parse(template)
    parts:List[ColumnElement] = []
    for prefix, colname, _, _ in format_tuples:
        if prefix != "":
            parts.append(literal(prefix))
        if colname:
            col = get_col(tab, colname)
            parts.append(col)
            if get_python_column_type(tab, colname) != str:
                col = sqlfunc.cast(col, sqltypes.VARCHAR)

    return sqlfunc.concat(*parts)

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

def get_python_column_type(tab:FromClause, col_name:str) -> Optional[Type[Any]]:
    #if isinstance(tab, FromClauseAlias):
    #    tab = tab.element
    col = get_col(cast(NamedFromClause, tab), col_name)
    return col.type.python_type

def expr_to_str(ex:ColumnElement):
    return str(ex.compile(compile_kwargs={"literal_binds": True}))

def same_expressions(ex1:ColumnElement, ex2:ColumnElement):
    return expr_to_str(ex1) == expr_to_str(ex2)

class NewR2rStore(R2RStore):

    pomaps_by_predicate: Dict[URIRef|None, List[Node]]
    metadata:MetaData

    def __init__(self, db: Engine, mapping: R2RMapping | None = None, base: str = "http://example.com/base/", configuration=None, identifier=None):
        super().__init__(db, mapping, base, configuration, identifier)
        self.pomaps_by_predicate = {}
        for pms in self.mapping.ppat_pomaps.values():
            found = False
            for pm in pms:
                for const in self.mapping.graph.objects(pm, AlternativePath(rr.predicate, SequencePath(rr.predicateMap, rr.constant))):
                    if isinstance(const, URIRef):
                        self.pomaps_by_predicate.setdefault(const, []).append(pm)
                        found = True
                    else:
                        raise ValueError("Non-URI predicate")
            if not found:
                raise NotImplementedError("TODO (2): Only constant predicates are supported in predicateObjectMaps")
        self.metadata = MetaData()
        self.metadata.reflect(db)

            
    def queryBGP(self, conn: Connection, bgp: BGP) -> SelectVarSubForm:

        q = queue.Queue[ProcessingState]()
        if len(bgp) == 0:
            raise ValueError("Empty BGPs are not supported")
        q.put(ProcessingState({}, {}, [], list(bgp)))
        resulting_states: List[ProcessingState] = []

        while not q.empty():
            st = q.get()

            for nst in self.process_next_triple(st):
                if not nst.triples:
                    resulting_states.append(nst)
                else:
                    q.put(nst)

        #TODO Not sure what this all is, figure it out later
        subforms: Dict[Variable, SubForm] = {}
        select_exprs: List[ColumnElement] = []

        if not resulting_states:
            raise ValueError(f"Failed to translate to SQL: {bgp}")
        if len(resulting_states) > 1:
            raise NotImplementedError("TODO (1): Multiple BGPs are not supported yet")
        
        rs = resulting_states[0]
        for i, (var, expr) in enumerate(rs.var_expressions.items()):
            select_exprs.append(expr)
            subforms[var] = SubForm([i], ExpressionTemplate.from_expr(expr).form)
        return SelectVarSubForm(select(*select_exprs).where(sql_and(*rs.wheres)), subforms)

    def process_next_triple(self, st: ProcessingState) -> Generator[ProcessingState, None, None]:
        mg = self.mapping.graph
        s, p, o = st.triples[0]
        assert s
        assert p 
        assert o
        if not isinstance(p, URIRef):
            raise NotImplementedError("TODO (2): Only URIRef predicates are supported")
        
        poms = self.pomaps_by_predicate.get(p, [])
        for pom in poms:
            for tm in mg.subjects(rr.predicateObjectMap, pom):
                row = st.rows.get(s, None)
                if not row:
                    tab = _get_table(mg, tm)
                    tab = self.metadata.tables[tab.name]
                    row = Row(subject=s, table=tab.alias("t"+str(len(st.rows))))
                    st = replace(st, rows={**st.rows, s: row})
                for st1 in self.match_node_to_term_map(s, tm, "S", st, row.table):
                    for st2 in self.match_node_to_term_map(o, pom, "O", st1, row.table):
                        for st3 in self.match_node_to_term_map(p, pom, "P", st2, row.table):
                            yield replace(st3, triples=st3.triples[1:])

    def match_node_to_term_map(self, node:Node, term_map:Node, position: LiteralType["S","P","O"], st:ProcessingState, 
                tab:NamedFromClause) -> Generator[ProcessingState, None, None]:
        mg = self.mapping.graph
        if position == "S":
            map_property = rr.subjectMap
            shortcut_property = rr.subject
        elif position == "P":
            map_property = rr.predicateMap
            shortcut_property = rr.predicate
        elif position == "O":
            map_property = rr.objectMap
            shortcut_property = rr.object
        else:
            raise ValueError("Invalid position: " + str(position))
        
        def match_variable(node:Variable, expr: ColumnElement) -> Generator[ProcessingState, None, None]:
            vex = st.var_expressions.get(node, None)
            if vex is not None:
                if same_expressions(expr,vex):
                    yield st
                else:
                    yield replace(st, wheres=st.wheres + [vex == expr])
            else:
                yield replace(st, var_expressions={**st.var_expressions, node: expr})

        for const in mg.objects(term_map, AlternativePath(shortcut_property, SequencePath(map_property, rr.constant))):
            if isinstance(node, Variable):
                yield from match_variable(node, literal(toPython(const)))
            elif const == node:
                yield st
            return # Only one term spec is allowed
        
        for tm in mg.objects(term_map, map_property):
            #is_iri = True
            #if position == "O" and (tm, rr.datatype, None) in mg or (tm, rr.language, None) in mg:
            #    is_iri = False
                
            for column in mg.objects(tm, rr.column):
                #if position == "O":
                #    is_iri = False

                colex = get_col(tab, str(column))
                if isinstance(node, Variable):
                    yield from match_variable(node, colex)
                else:
                    yield replace(st, wheres=st.wheres + [colex == toPython(node)])
                return # Only one term spec is allowed

            for template in mg.objects(tm, rr.template):
                expr = format_template(str(template), tab)
                if isinstance(node, Variable):
                    yield from match_variable(node, expr)
                else:
                    #TODO check IRI vs literal
                    col_vals = parse_with_template(str(node), str(template))
                    if col_vals:
                        wheres = []
                        for col, val in col_vals.items():
                            ptype = get_python_column_type(tab, col)
                            if ptype:
                                pval = ptype(val)
                            else:
                                pval = val #TODO ???
                            wheres.append(get_col(tab, col) == pval)
                        yield replace(st, wheres=st.wheres + wheres)
                    # The old brute-force way:
                    #yield replace(st, wheres=st.wheres + [expr == toPython(node)])
                return

            for parent in mg.objects(tm, rr.parentTriplesMap):
                raise NotImplementedError("TODO (1): Parent triples maps not implemented")
                for st1 in self.match_node_to_term_map(node, parent, position, st, tab):
                    yield st1
                return