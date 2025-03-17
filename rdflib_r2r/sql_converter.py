from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Generator, List, Literal as LiteralType, Mapping, Optional, Type, cast
from rdflib import RDF, Graph, IdentifiedNode, Literal, URIRef, Variable, BNode
from rdflib.term import Node
from rdflib.paths import Path, AlternativePath, SequencePath, InvPath
from sqlalchemy import Alias, Dialect, MetaData, Select
from sqlalchemy.engine import Engine
from sqlalchemy.sql import ColumnElement, select, literal
from sqlalchemy.sql.selectable import NamedFromClause, FromClause
from rdflib_r2r.query_conversions import QueryConversions
from rdflib_r2r.r2r_mapping import _get_table, rr, toPython
from rdflib_r2r.conversion_utils import equal, expr_to_str, format_template, iter_opt, parse_with_template, results_union, sql_and, sql_pretty
from rdflib_r2r.types import BGP, SPARQLVariable, SQLQuery, SearchQuery
import queue

@dataclass(frozen=True, eq=True, kw_only=True)
class Row:
    subject: Node
    table: NamedFromClause

@dataclass(frozen=True, eq=True, kw_only=True)
class VariableExpression:
    expression: ColumnElement
    template: str|None

def make_short_alias(name:str):
    alias = ''
    prev = None
    for c in name:
        if c.isalpha():
            if prev is None or c.isupper() and not prev.isupper():
                alias += c.lower()
            prev = c
        else:
            prev = None
    return alias

@dataclass(frozen=True, eq=True, kw_only=True)
class ProcessingState:
    store:"SQLConverter"
    rows: Dict[Node,Row]
    var_expressions: Dict[SPARQLVariable, ColumnElement]
    wheres: List[ColumnElement[bool]]
    triples: List[SearchQuery]

    def ensure_row(self, s:Node, triple_map:Node) -> "Optional[ProcessingState]":
        row = self.rows.get(s, None)
        tab = _get_table(self.store.mapping_graph, triple_map)
        tab = self.store.metadata.tables[tab.name]
        if not row:
            if isinstance(s, Variable):
                alias = str(s)

                key = (alias, tab.name)
                if key in self.store.current_project.named_tables:
                    atab = self.store.current_project.named_tables[key]
                else:
                    atab = tab.alias(alias)
                    self.store.current_project.named_tables[key] = atab
            else:
                alias0 = make_short_alias(tab.name)
                alias = alias0
                i = 0
                current_aliases = { row.table.name for row in self.rows.values() }
                while alias in current_aliases or alias in self.store.current_project.named_vars:
                    alias = alias0 + str(i)
                    i += 1
                atab = tab.alias(alias)

            row = Row(subject=s, table=atab)
            return replace(self, rows={**self.rows, s: row})
        elif cast(Alias, row.table).original != tab:
            return None
        else:
            return self

def get_python_column_type(tab:FromClause, col_name:str) -> Optional[Type[Any]]:
    #if isinstance(tab, FromClauseAlias):
    #    tab = tab.element
    col = tab.c[col_name]
    return col.type.python_type

def same_expressions(ex1:ColumnElement, ex2:ColumnElement):
    return expr_to_str(ex1) == expr_to_str(ex2)

def n3(nd:Node|Path|None|tuple[Node|None,Node|Path|None,Node|None], g:Graph|None = None):
    if isinstance(nd, tuple):
        s,p,o = nd
        return "("+" ".join([n3(s,g), n3(p,g), n3(o,g)])+")"
    if isinstance(nd, (IdentifiedNode,Variable,Path)):
        return nd.n3(g.namespace_manager if g is not None else None)
    else:
        return str(nd)
    
def substitute_first_triple(triples: BGP, replacement_triples: List[SearchQuery], exclude_func:Callable[[Path],bool]) -> Generator[List[SearchQuery], None, None]:
    replacement_triples += triples[1:]
    yield from resolve_paths_in_triples(replacement_triples, exclude_func)

def resolve_paths_in_triples(triples: BGP,  exclude_func: Callable[[Path],bool]) -> Generator[List[SearchQuery], None, None]:
    if not triples:
        yield []
        return
    
    t0 = triples[0]
    p = t0[1]
    if isinstance(p, SequencePath):
        if exclude_func(p):
            for ts in resolve_paths_in_triples(triples[1:], exclude_func):
                yield [t0] + ts
            return
        replacement_triples = []
        next_subj = t0[0]
        for p1 in p.args[:-1]:
            obj = BNode()
            replacement_triples.append((next_subj, p1, obj))
            next_subj = obj
        replacement_triples.append((next_subj, p.args[-1], t0[2]))
        yield from substitute_first_triple(triples, replacement_triples, exclude_func)
    elif isinstance(p, AlternativePath):
        for p1 in p.args:
            yield from substitute_first_triple(triples, [(t0[0], p1, t0[2])], exclude_func)
    elif isinstance (p, InvPath):
        if isinstance(t0[2], (URIRef, BNode, Variable)):
            yield from substitute_first_triple(triples, [(t0[2], p.arg, t0[0])], exclude_func)
        else:
            raise ValueError("Literals not supported as inverse path objects")
    elif isinstance (p, Path):
        raise NotImplementedError("Unsupported path type: " + str(p))
    else:
        for ts in resolve_paths_in_triples(triples[1:], exclude_func):
            yield [t0] + ts

class SQLConverter(QueryConversions):

    pomaps_by_predicate: Dict[URIRef|Literal|None, List[Node]]
    metadata:MetaData
    dialect:Dialect
    mapping_graph:Graph

    def __init__(self, db: Engine, mapping_graph: Graph):
        super().__init__()
        self.mapping_graph = mapping_graph
        self.pomaps_by_predicate = {}
        for pm in self.mapping_graph.objects(predicate=rr.predicateObjectMap, unique=True):
            found = False
            for const in self.mapping_graph.objects(pm, AlternativePath(rr.predicate, SequencePath(rr.predicateMap, rr.constant))):
                if isinstance(const, URIRef):
                    self.pomaps_by_predicate.setdefault(const, []).append(pm)
                    found = True
                else:
                    raise ValueError("Non-URI predicate")
            if not found:
                raise NotImplementedError("TODO (2): Only constant predicates are supported in predicateObjectMaps")

        self.add_synthetic_chain_triple_maps()

        self.metadata = MetaData()
        self.metadata.reflect(db)
        self.dialect = db.dialect

    def getSQL(self, sparqlQuery, base:str|None=None, initNs:Mapping[str, Any] | None={}):
        from rdflib.plugins.sparql.parser import parseQuery
        from rdflib.plugins.sparql.algebra import translateQuery

        parsetree = parseQuery(sparqlQuery)
        queryobj = translateQuery(parsetree, base, initNs)
        sql_query = self.queryPart(queryobj.algebra)
        return sql_pretty(sql_query, dialect=self.dialect)        
            
    def queryBGP(self, bgp: BGP) -> SQLQuery:
        q = queue.Queue[ProcessingState]()

        def should_leave_path_alone(p:Path) -> bool:
            return Literal(p.n3(self.mapping_graph.namespace_manager)) in self.pomaps_by_predicate
        
        for rbgp in resolve_paths_in_triples(bgp, should_leave_path_alone):
            q.put(ProcessingState(store=self, rows={}, var_expressions={}, wheres=[], triples=rbgp))
        resulting_states: List[ProcessingState] = []

        while not q.empty():
            st = q.get()
            if not st.triples:
                resulting_states.append(st)
            else:
                for nst in self.process_next_triple(st):
                    if not nst.triples:
                        resulting_states.append(nst)
                    else:
                        q.put(nst)

        if not resulting_states:
            raise ValueError(f"Failed to translate to SQL: { [n3(t,self.mapping_graph) for t in bgp]}")
        
        query_elements:List[Select] = []
        for rs in resulting_states:
            select_exprs = [ expr.label(str(var)) for var, expr in rs.var_expressions.items() if isinstance(var,Variable) ]
            result = select(*select_exprs)
            if rs.wheres:
                result = result.where(sql_and(*rs.wheres))
            query_elements.append(result)

        if len(query_elements) == 1:
            self.current_project.add_variables_to_columns(query_elements[0])
        # I believe we don't actually need to handle multiple results, because in all situations when we have them
        # the composite query wil be wrapped in a Select before anything else interesting happens.
        return results_union(query_elements)
    
    def process_next_triple(self, st: ProcessingState) -> Generator[ProcessingState, None, None]:
        mg = self.mapping_graph
        s, p, o = st.triples[0]
        assert s
        assert p 
        assert o
        if not isinstance(p, (URIRef,Path)):
            raise NotImplementedError("TODO (2): Only URIRef or Path predicates are supported")

        if p == RDF.type:
            if isinstance(o, SPARQLVariable):
                raise NotImplementedError("TODO (2): retrieving types not supported")
            for sm in mg.subjects(rr['class'], o):
                for tm in mg.subjects(rr.subjectMap, sm):
                    for rst in iter_opt(st.ensure_row(s, tm)):
                        row = rst.rows[s]
                        for st1 in self.match_node_to_term_map(s, tm, "S", rst, row.table):
                            yield replace(st1, triples=st1.triples[1:])

        pkey = Literal(p.n3(self.mapping_graph.namespace_manager)) if isinstance(p, Path) else p
        poms = self.pomaps_by_predicate.get(pkey, [])
        for pom in poms:
            for tm in mg.subjects(rr.predicateObjectMap, pom):
                for stR in iter_opt(st.ensure_row(s,tm)):
                    row = stR.rows[s]
                    for st1 in self.match_node_to_term_map(s, tm, "S", stR, row.table):
                        for st2 in self.match_node_to_term_map(o, pom, "O", st1, row.table):
                            for st3 in self.match_node_to_term_map(p, pom, "P", st2, row.table):
                                yield replace(st3, triples=st3.triples[1:])

    def match_node_to_term_map(self, node:Node|Path, term_map:Node, position: LiteralType["S","P","O"], st:ProcessingState, 
                tab:NamedFromClause) -> Generator[ProcessingState, None, None]:
        mg = self.mapping_graph
        if isinstance(node, Path):
            if position != "P":
                return
            for cs in mg.objects(term_map, SequencePath(rr.predicateMap, rr.chainStr)):
                if node.n3(mg.namespace_manager) == str(cs):
                    yield st
            return

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
        
        def match_variable(node:SPARQLVariable, expr: ColumnElement) -> Generator[ProcessingState, None, None]:
            vex = st.var_expressions.get(node, None)
            if vex is not None:
                if same_expressions(expr,vex):
                    yield st
                else:
                    yield replace(st, wheres=st.wheres + list(equal(vex, expr)))
            else:
                yield replace(st, var_expressions={**st.var_expressions, node: expr})

        for const in mg.objects(term_map, AlternativePath(shortcut_property, SequencePath(map_property, rr.constant))):
            if isinstance(node, SPARQLVariable):
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

                colex = tab.c[str(column)]
                if isinstance(node, SPARQLVariable):
                    yield from match_variable(node, colex)
                else:
                    yield replace(st, wheres=st.wheres + list(equal(colex, toPython(node))))
                return # Only one term spec is allowed

            for template in mg.objects(tm, rr.template):
                expr = format_template(str(template), tab)
                if isinstance(node, SPARQLVariable):
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
                            wheres.append(tab.c[col] == pval)
                        yield replace(st, wheres=st.wheres + wheres)
                    # The old brute-force way:
                    #yield replace(st, wheres=st.wheres + [expr == toPython(node)])
                return

            for parent_triple_map in mg.objects(tm, rr.parentTriplesMap):
                for jrst in iter_opt(st.ensure_row(node, parent_triple_map)):
                    wheres: List[ColumnElement[bool]] = []
                    jrow = jrst.rows[node]
                    for join in mg.objects(tm, rr.joinCondition):
                        childColumn = mg.value(join, rr.child)
                        assert childColumn
                        parentColumn = mg.value(join, rr.parent)
                        assert parentColumn
                        wheres.append(tab.c[str(childColumn)] == jrow.table.c[str(parentColumn)])

                    jrst = replace(jrst, wheres = jrst.wheres + wheres)
                    yield from self.match_node_to_term_map(node, parent_triple_map, "S", jrst, jrow.table)

    def add_synthetic_chain_triple_maps(self):

        def add_synthetic_chain_tm(tm, pred1, pred2, col_name, inv):
            pomap = BNode()
            self.mapping_graph.add((tm, rr.predicateObjectMap, pomap))
            ppmap = BNode()
            assert isinstance(pred1, URIRef)
            assert isinstance(pred2, URIRef)
            chain_str = Literal(SequencePath(InvPath(pred1) if inv else pred1, pred2).n3(self.mapping_graph.namespace_manager))
            self.mapping_graph.add((ppmap, rr.chainStr, chain_str))
            self.mapping_graph.add((pomap, rr.predicateMap, ppmap))
            omap = BNode()
            self.mapping_graph.add((omap, rr.column, col_name))
            self.mapping_graph.add((pomap, rr.objectMap, omap))
            self.mapping_graph.add((tm, rr.predicateObjectMap, pomap))
            self.pomaps_by_predicate.setdefault(chain_str, []).append(pomap)

        for col_name, tm1, pred1, pred2  in  self.mapping_graph.query( # type: ignore
                f'''
                select ?col_name ?tm1 ?pred1 ?pred2  {{
                    ?pomap rr:objectMap [
                            rr:joinCondition [ rr:child ?col_name; rr:parent ?id_col ];
                            rr:parentTriplesMap ?tm2
                        ] ;
                        (rr:predicate | rr:predicateMap / rr:constant) ?pred1.
                    
                    ?tm1 rr:predicateObjectMap ?pomap.

                    ?tm2 rr:predicateObjectMap [
                        (rr:predicate | rr:predicateMap / rr:constant) ?pred2;
                        rr:objectMap [ rr:column ?id_col ]
                    ]
                }}
                '''
                ):
            
            add_synthetic_chain_tm(tm1, pred1, pred2, col_name, False)    

        for col_name, tm2, pred1, pred2 in  self.mapping_graph.query( # type: ignore
                f'''
                select ?col_name ?tm2 ?pred1 ?pred2 {{
                    ?pomap rr:objectMap [
                            rr:joinCondition [ rr:child ?id_col; rr:parent ?col_name ];
                            rr:parentTriplesMap ?tm2
                        ] ;
                        (rr:predicate | rr:predicateMap / rr:constant) ?pred2.
                    
                    ?tm1 rr:predicateObjectMap ?pomap, [
                        (rr:predicate | rr:predicateMap / rr:constant) ?pred1;
                        rr:objectMap [ rr:column ?id_col ]
                    ].

                    ?tm2 rr:logicalTable / rr:tableName ?table_name.
                }}
                '''
                ):
            add_synthetic_chain_tm(tm2, pred2, pred1, col_name, True)    
