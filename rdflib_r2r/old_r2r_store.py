from collections import Counter
import functools
import logging
import operator
import re
from string import Formatter
from types import NoneType
from typing import Any, Generator, Iterable, Iterator, List, Optional, Tuple, Union, cast
from rdflib import RDF, XSD, Graph, Variable
from rdflib.term import Node, URIRef, Literal
from rdflib_r2r.expr_template import ExpressionTemplate
from rdflib_r2r.r2r_mapping import _get_table
from rdflib_r2r.r2r_store import GenerativeSelectSubForm, R2RStore, SelectSubForm, SelectVarSubForm, SubForm, sql_safe, rr
import sqlalchemy
import sqlalchemy.sql.operators
from sqlalchemy import MetaData, select, text, null, literal_column, literal, TableClause, Subquery, Table, ClauseElement, Function
from sqlalchemy import union_all, or_ as sql_or, and_ as sql_and
from sqlalchemy import schema as sqlschema, types as sqltypes, func as sqlfunc
import sqlalchemy.sql as sql
from sqlalchemy.sql.expression import ColumnElement, GenerativeSelect, Select, ColumnClause
from sqlalchemy.sql.selectable import ScalarSelect, CompoundSelect, NamedFromClause
from sqlalchemy.engine import Engine, Connection
from rdflib_r2r.types import BGP, Triple
from rdflib.util import from_n3


class OldR2RStore(R2RStore):

    @classmethod
    def _term_map_colforms(
        cls, graph: Graph, dbtable: NamedFromClause | Subquery, parent: Node, wheres: List[ColumnElement[bool]], 
            mapper: Node, shortcut:Node, obj=False
    ) -> Generator[tuple[ExpressionTemplate, Table | NamedFromClause | Subquery], Any, Any]: 
        """For each Triples Map, yield a expression template containing table columns.

        Args:
            graph (Graph): The RDF2RDB mapping graph
            dbtable (TableClause): The database table for this mapping
            parent (Identifier): The parent of this mapping
            wheres (Iterable[BooleanClause]): SQL clauses for restriction
            mapper (URIRef): Mapping predicate
            shortcut (URIRef): Mapping shortcut predicate
            obj (bool): Whether this term is in the object position. Defaults to False.

        Yields:
            ColForm: Expression Templates
        """
        if graph.value(parent, shortcut):
            # constant shortcut properties
            for const in graph.objects(parent,shortcut):
                yield ExpressionTemplate([f"'{const.n3()}'"], []), dbtable
        elif graph.value(parent, mapper):
            for tmap in graph.objects(parent,mapper):
                if graph.value(tmap, rr.constant):
                    # constant value
                    for const in graph.objects(tmap, rr.constant):
                        yield ExpressionTemplate([f"'{const.n3()}'"], []), dbtable
                else:
                    termtype = graph.value(tmap, rr.termType) or rr.IRI
                    if graph.value(tmap, rr.column):
                        colname = graph.value(tmap, rr.column)
                        assert colname
                        colform = ExpressionTemplate.from_expr(cls._get_col(dbtable, str(colname)))
                        if obj:
                            # for objects, the default term type is Literal
                            termtype = graph.value(tmap, rr.termType) or rr.Literal
                    elif graph.value(tmap, rr.template):
                        template = graph.value(tmap, rr.template)
                        colform = ExpressionTemplate.from_template(
                            dbtable, template, irisafe=(termtype == rr.IRI)
                        )
                    elif graph.value(tmap, rr.parentTriplesMap):
                        # referencing object map
                        ref = graph.value(tmap, rr.parentTriplesMap)
                        assert ref
                        ptable = _get_table(graph, ref)
                        ptable = ptable.alias(f"{ptable.name}_ref")
                        # push the where clauses into the subquery
                        joins = wheres
                        for join in graph.objects(tmap, rr.joinCondition):
                            # child column and parent column
                            ccol = f'"{dbtable.name}".{graph.value(join, rr.child)}'
                            pcol = f'"{ptable.name}".{graph.value(join, rr.parent)}'
                            joins.append(literal_column(ccol) == literal_column(pcol))
                        referenced_colforms = cls._term_map_colforms(
                            graph, ptable, ref, [], rr.subjectMap, rr.subject
                        )
                        for colform, table in referenced_colforms:
                            cols = [c.label(None) for c in colform.cols]
                            colform = ExpressionTemplate(colform.form, cols)
                            yield colform, table
                        continue
                    else:
                        # TODO: replace with RDB-specific construct (postgresql?)
                        rowid = literal_column(f'"{dbtable.name}".rowid').cast(
                            sqltypes.VARCHAR
                        )
                        form = ["_:" + dbtable.name.replace('_ref','') + "#", None]
                        yield ExpressionTemplate(form, [rowid]), dbtable
                        continue

                    if termtype == rr.IRI:
                        form = ["<"] + list(colform.form) + [">"]
                        yield ExpressionTemplate(form, colform.cols), dbtable
                    elif termtype == rr.BlankNode:
                        yield ExpressionTemplate((["_:"] + list(colform.form)), colform.cols), dbtable
                    elif obj:
                        if graph.value(tmap, rr.language):
                            lang = graph.value(tmap, rr.language)
                            cols = [
                                sqlfunc.cast(c, sqltypes.VARCHAR) for c in colform.cols
                            ]
                            form = ['"'] + list(colform.form) + ['"@' + str(lang)]
                            yield ExpressionTemplate(form, cols), dbtable
                        elif graph.value(tmap, rr.datatype):
                            dtype = graph.value(tmap, rr.datatype)
                            assert isinstance(dtype, URIRef)
                            cols = [
                                sqlfunc.cast(c, sqltypes.VARCHAR) for c in colform.cols
                            ]
                            form = ['"'] + list(colform.form) + ['"^^' + dtype.n3()]
                            yield ExpressionTemplate(form, cols), dbtable
                        else:
                            # keep original datatype
                            yield colform, dbtable
                    else:
                        # not a real literal
                        yield ExpressionTemplate.from_expr(literal_column("'_:'")), dbtable

    def _triplesmap_select(self, metadata:MetaData, tmap:Node, 
            pattern:tuple[Node | None, Node | None, Node | None]) -> Generator[SelectSubForm,None,None]:
        mg = self.mapping.graph

        dbtable = _get_table(mg, tmap)
        if metadata:
            dbtable = metadata.tables.get(dbtable.name, dbtable)

        qs, qp, qo = pattern
        sfilt = self.mapping.get_node_filter(qs, self.mapping.spat_tmaps)
        pfilt = self.mapping.get_node_filter(qp, self.mapping.ppat_pomaps)
        ofilt = self.mapping.get_node_filter(qo, self.mapping.opat_pomaps)

        swhere = []
        if not (None in sfilt):
            if not (tmap in sfilt):
                return
            else:
                swhere = sfilt[tmap]

        ss = self._term_map_colforms(
            mg, dbtable, tmap, swhere, rr.subjectMap, rr.subject
        )
        scolform, stable = next(ss)
        s_map = mg.value(tmap, rr.subjectMap)
        assert s_map

        gcolforms = list(
            self._term_map_colforms(mg, dbtable, s_map, [], rr.graphMap, rr.graph)
        ) or [(ExpressionTemplate.null(), dbtable)]

        # Class Map
        if (not pfilt) or (None in pfilt) or (RDF.type == qp):
            for c in mg.objects(s_map, rr["class"]):
                pcolform = ExpressionTemplate([f"'{RDF.type.n3()}'"], [])
                ocolform = ExpressionTemplate([f"'{c.n3()}'"], [])
                # no unsafe IRI because it should be defined to be safe
                if (qo is not None) and (qo != c):
                    continue
                for gcolform, gtable in gcolforms:
                    subforms, cols = ExpressionTemplate.to_subforms_columns(
                        scolform, pcolform, ocolform, gcolform
                    )
                    tables = set([stable, gtable])
                    query = select(*cols).select_from(*tables)
                    if swhere:
                        query = query.where(*swhere)
                    yield SelectSubForm(query, subforms)

        # Predicate-Object Maps
        pomaps = set(mg.objects(tmap, rr.predicateObjectMap))
        if not (None in pfilt):
            pomaps &= set(pfilt)
        if not (None in ofilt):
            pomaps &= set(ofilt)

        for pomap in pomaps:
            pwhere = pfilt.get(pomap) or []
            pcolforms = self._term_map_colforms(
                mg, dbtable, pomap, pwhere, rr.predicateMap, rr.predicate
            )
            owhere = ofilt.get(pomap) or []
            ocolforms = list(
                self._term_map_colforms(
                    mg, dbtable, pomap, owhere, rr.objectMap, rr.object, True
                )
            )
            gcolforms = list(
                self._term_map_colforms(mg, dbtable, pomap, [], rr.graphMap, rr.graph)
            ) or [(ExpressionTemplate.null(), dbtable)]
            for pcolform, ptable in pcolforms:
                pstr = "".join([ str(fe) for fe in pcolform.form if fe ])
                if (qp is not None) and pstr[1:-1] != qp.n3():
                    # Filter out non-identical property patterns
                    continue
                for ocolform, otable in ocolforms:
                    for gcolform, gtable in gcolforms:
                        where = swhere + pwhere + owhere
                        subforms, cols = ExpressionTemplate.to_subforms_columns(
                            scolform, pcolform, ocolform, gcolform
                        )
                        tables = set([stable, ptable, otable, gtable])
                        query = select(*cols).select_from(*tables)
                        if where:
                            query = query.where(*where)
                        yield SelectSubForm(query, subforms)

    @staticmethod
    def col_n3(dbcol):
        """Cast column to n3"""
        if isinstance(dbcol.type, sqltypes.DATE):
            dt = XSD.date.n3()
            n3col = '"' + sqlfunc.cast(dbcol, sqltypes.VARCHAR) + ('"^^' + dt)
            #XXX Not sure what this is supposed to do, but ColumnElement has no such attribute.
            #n3col.original = dbcol
            return n3col
        if isinstance(dbcol.type, sqltypes.DATETIME) or isinstance(
            dbcol.type, sqltypes.TIMESTAMP
        ):
            dt = XSD.dateTime.n3()
            value = sqlfunc.replace(sqlfunc.cast(dbcol, sqltypes.VARCHAR), " ", "T")
            n3col = '"' + value + ('"^^' + dt)
            #n3col.original = dbcol
            return n3col
        if isinstance(dbcol.type, sqltypes.BOOLEAN):
            dt = XSD.boolean.n3()
            value = sql.expression.case({1: "true", 0: "false"}, value=dbcol)
            n3col = '"' + value + ('"^^' + dt)
            #n3col.original = dbcol
            return n3col
        if isinstance(dbcol.type, sqltypes.INT):
            dt = XSD.integer.n3() # if this can run
            value = sqlfunc.cast(dbcol, sqltypes.VARCHAR)
            n3col = '"' + value + ('"^^' + dt)
            #n3col.original = dbcol
            return n3col
        ## TODO: create n3 literal for integers!

        return dbcol

    def union_spog_querysubforms(self, *queryforms) -> GenerativeSelectSubForm:
        queries = []
        # Make expressions from ColForms
        for query, (s,p,o,g) in queryforms:
            cols = list(query.exported_columns)
            onlycols = []
            for subform, name in zip((s,p,o,g), "spog"):
                col = ExpressionTemplate.from_subform(cols, *subform).expr()
                onlycols.append(col.label(name))
            queries.append(query.with_only_columns(*onlycols))
        subforms = [SubForm([i], (None,)) for i in range(4)]  # spog

        # If the object columns have different datatypes, cast them to n3 strings
        # WARNING: In most cases, this should be fine but it might mess up!
        _, _, o_cols, *_ = zip(*[q.exported_columns for q in queries])
        kwargs = lambda c: tuple((k, v) for k, v in vars(c).items() if k[0] != "_")
        o_types = set((c.type.__class__, kwargs(c.type)) for c in o_cols)
        logging.warn(f"UNION types: {o_types}")

        if len(o_types) > 1:
            for qi, query in enumerate(queries):
                s, p, o, g = query.exported_columns
                logging.warn(f"object column: {o} type: {o.type}")
                queries[qi] = query.with_only_columns(*[s, p, self.col_n3(o), g])
        
        return GenerativeSelectSubForm(union_all(*queries), subforms)

    def queryPattern(
        self, metadata:MetaData, pattern:tuple[Node | None, Node | None, Node | None], 
        restrict_tmaps:set[URIRef]|None = None
    ) -> tuple[GenerativeSelect, List[SubForm]]:
        """Make a set of SubForms for a GenerativeSelect query from a triple pattern

        Args:
            metadata: Database Metadata object
            pattern: Triple pattern
            restrict_tmaps: 

        Returns:
            GenerativeSelectSubForm: _description_
        """
        querysubforms: List[SelectSubForm] = []
        # Triple Maps produce select queries
        for tmap in self.mapping.graph.subjects(RDF.type, rr.TriplesMap): # TODO: get rid of rr.TriplesMap because it might not be explicitly stated in the mapping
            if restrict_tmaps is not None and (tmap not in restrict_tmaps):
                mg = self.mapping.graph
                refs = set(
                    ref
                    for pomap in mg.objects(tmap, rr.predicateObjectMap)
                    for omap in mg.objects(pomap, rr.objectMap)
                    for ref in mg.objects(omap, rr.parentTriplesMap)
                )
                if not any(t in refs for t in restrict_tmaps):
                    continue
            querysubforms += list(self._triplesmap_select(metadata, tmap, pattern))

        if len(querysubforms) > 1:
            return self.union_spog_querysubforms(*querysubforms)
        elif querysubforms:
            return querysubforms[0]
        else:
            raise Exception(f"Didn't get tmaps for {pattern} from {restrict_tmaps}!")
            
    def triples(self, triple_pattern:Tuple[Node|None,Node|None,Node|None], context=None) -> Generator[
            Tuple[Triple, Iterator[Optional[Graph]]],None,None]:
        """Search for a triple pattern in a DB mapping.

        Args:
            pattern: The triple pattern (s, p, o) to search.
            context: The query execution context.

        Returns:
            An iterator that produces RDF triples matching the input triple pattern.
        """
        def nonvar(n): 
            return n if not isinstance(n, Variable) else None
        triple_pattern = (nonvar(triple_pattern[0]), nonvar(triple_pattern[1]), nonvar(triple_pattern[2]))

        result_count = 0
        with self.db.connect() as conn:
            metadata = MetaData()
            metadata.reflect(self.db)

            query, (s,p,o,g) = self.queryPattern(metadata, triple_pattern)
            if isinstance(query, CompoundSelect):
                query = query.subquery()
            # logging.warn('query:' + str(query))
            # logging.warn('subforms:' + str(s,p,o,g))
            cols = getattr(query, "exported_columns", query.c)
            # logging.war('cols:' + str(list(cols)))
            onlycols = []
            for subform, colname in zip((s,p,o,g), "spog"):
                col = ExpressionTemplate.from_subform(cols, *subform).expr()
                onlycols.append(col.label(colname))

            if isinstance(query, Select):
                query = query.with_only_columns(*onlycols)
            else:
                query = select(*onlycols)

            # logging.warn('final query:' + sql_pretty(query))
            rows = list(conn.execute(query))
            for s, p, o, g in rows:
                gnode = from_n3(g)
                snode = self.make_node(s)
                pnode = self.make_node(p)
                onode = self.make_node(o)
                if (snode is None) or (onode is None):
                    continue

                result = snode, pnode, onode
                if any(r is None for r in result):
                    logging.warn(f"none in result: {result}")
                result_count += 1
                yield result, gnode

        ns = self.mapping.graph.namespace_manager
        patstr = " ".join((n.n3(ns) if n else "_") for n in triple_pattern)
        # logging.warn(f"pattern: {patstr}, results: {result_count}")

    ###### SPARQL #######

    def queryBGP(self, conn: Connection, bgp: BGP) -> SelectVarSubForm:
        """Generate a Basic Graph Pattern subquery

        Args:
            conn: SQLAlchemy database connection
            bgp: Basic Graph Pattern

        Returns:
            GenerativeSelect: Subquery that matches the BGP
        """
        bgp = set(bgp)

        metadata = MetaData()
        metadata.reflect(self.db)

        # Optimize DB table restrictions in queries
        mg = self.mapping.graph

        # Maps variables that are subjects to the triple maps that apply to them
        restrict_tmaps:dict[Variable,set[URIRef]] = {}

        # Loop through the triples in the graph pattern
        for qs, qp, qo in bgp:
            if isinstance(qs, Variable):
                restriction = set()
                # Find triple map restrictions based on types
                if (not isinstance(qo, Variable)) and (qp == RDF.type):
#                     for tmap in mg[: RDF.type : rr.TriplesMap]: # loop over subjects (== mg.subjects(RDF.type, rr.TriplesMap))
#                         sm = mg.value(tmap, rr.subjectMap) # get object (== next(mg.objects(tmap, rr.subjectMap)))
#                         if qo in mg[sm : rr["class"]]: # check if qo is in objects ( == (qo in mg.objects(sm, rr.class)) if it wasn't a syntax error) 
#                             restriction.add(tmap)
                    for sm in mg.subjects(rr["class"], qo):
                       tmap = next(mg.subjects(rr.subjectMap, sm))
                       restriction.add(tmap)
                # Find triple map restrictions based on predicates
                for pomap in mg.subjects(rr.predicate, qp):
                    # Other triple maps that share this pred-obj map
                    for tmap in mg.subjects(rr.predicateObjectMap, pomap):
                        restriction.add(tmap)
                    # Referenced triple maps
                    for omap in mg.objects(pomap, rr.objectMap):
                        for tmap in mg.objects(omap, rr.parentTriplesMap):
                            # recursive ??
                            restriction.add(tmap)
                if restriction:
                    if qs in restrict_tmaps:
                        restrict_tmaps[qs] &= restriction  # intersect per pattern
                    else:
                        restrict_tmaps[qs] = restriction

        # Collect queries and associated query variables; collect simple table selects
        def novar(n):
            return n if not isinstance(n, Variable) else None
        query_varsubforms = []
        table_varcolforms = {}
        for qs, qp, qo in bgp:
            # The triple maps for the subject variable if any
            restriction = restrict_tmaps.get(qs) if isinstance(qs, Variable) else None
            pat = novar(qs), novar(qp), novar(qo)

            # Keep track of which columns belong to which query variable
            pat_query, subforms = self.queryPattern(metadata, pat, restriction)
            if isinstance(pat_query, Select):
                # Restrict selected terms from pattern query to query variables
                cols = list(pat_query.exported_columns)
                qvar_colform = [
                    (q, ExpressionTemplate.from_subform(cols, *subform))
                    for q, subform in zip((qs, qp, qo), subforms)
                    if isinstance(q, Variable)
                ]
                if len(pat_query._from_obj) == 1:
                    # Single table, so try to merge shared-subject terms
                    table = pat_query._from_obj[0], pat_query.whereclause
                    table_varcolforms.setdefault(table, set()).update(qvar_colform)
                else:
                    qvars, colforms = zip(*qvar_colform)
                    subforms, allcols = ExpressionTemplate.to_subforms_columns(*colforms)
                    pat_query = pat_query.with_only_columns(*allcols)
                    qvar_subform = zip(qvars, subforms)
                    query_varsubforms.append((pat_query, qvar_subform))
            else:
                qvar_subform = [
                    (q, subform)
                    for q, subform in zip((qs, qp, qo), subforms)
                    if isinstance(q, Variable)
                ]
                query_varsubforms.append((pat_query, qvar_subform))

        # Merge simple select statements on same table
        for (table, where), var_colforms in table_varcolforms.items():
            qvars, colforms = zip(*dict(var_colforms).items())
            subform, allcols = ExpressionTemplate.to_subforms_columns(*colforms)
            query = select(*allcols).select_from(table)
            if where is not None:
                query = query.where(where)
            query_varsubforms.append((query, zip(qvars, subform)))

        # Collect colforms per variable
        var_colforms = {}
        for query, var_subform in query_varsubforms:
            subquery = query.subquery()
            incols = list(subquery.c)
            for var, (idx, form) in var_subform:
                cols = [incols[i] for i in idx]
                var_colforms.setdefault(var, []).append(ExpressionTemplate(form, cols))

        # Simplify colform equalities
        colforms = [cfs[0] for cfs in var_colforms.values()]
        subforms, allcols = ExpressionTemplate.to_subforms_columns(*colforms)
        where = [eq for cs in var_colforms.values() for eq in ExpressionTemplate.equal(*cs)]
        return SelectVarSubForm(select(*allcols).where(*where), dict(zip(var_colforms, subforms)))
