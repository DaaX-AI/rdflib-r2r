"""
The mapped SQL database as an RDF graph store object

.. note::
   TODO:
   
   * delay the UNION creation longer, so that we can use colforms to filter them

    * This could be complicated because you'd need to explode the BGPs 
      because each triple pattern becomes a set of selects, which you'd need to join



"""
from dataclasses import dataclass
from os import linesep
from types import NoneType
from typing import Any, Generator, Generic, Iterator, Optional, Iterable, List, Tuple, Union, Dict, NamedTuple, cast, TypeVar
import logging
import functools
import operator
import base64
from decimal import Decimal
import re
from string import Formatter
from collections import Counter
import math

from rdflib import Graph, URIRef, Literal, BNode, Variable
from rdflib.namespace import RDF, XSD, Namespace
from rdflib.store import Store
from rdflib.util import from_n3
from rdflib.term import Node
from rdflib.plugins.sparql.parserutils import CompValue
import sqlalchemy
import sqlalchemy.sql.operators
from sqlalchemy import MetaData, select, text, null, literal_column, literal, TableClause, Subquery, Table, ClauseElement, Function
from sqlalchemy import union_all, or_ as sql_or, and_ as sql_and
from sqlalchemy import schema as sqlschema, types as sqltypes, func as sqlfunc
import sqlalchemy.sql as sql
from sqlalchemy.sql.expression import ColumnElement, GenerativeSelect, Select, ColumnClause
from sqlalchemy.sql.selectable import ScalarSelect, CompoundSelect, NamedFromClause

from sqlalchemy.engine import Engine, Connection

from rdflib_r2r.types import Triple, BGP
from rdflib_r2r.r2r_mapping import R2RMapping, _get_table, iri_safe, iri_unsafe, toPython

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

class ColForm:
    """A template object for creating SQL expressions that represent RDF nodes.

    Because the columns are separated from the template, this allows for efficient
    joins, filters, and aggregates.

    The ``form`` and ``cols`` list are combined by replacing non-strings in ``form`` by
    elements of ``cols`` sequentially.

    >>> from sqlalchemy import MetaData, Table, Column, Integer, String
    >>> metadata_obj = MetaData()
    >>> students_table = Table(
    >>>     "students",
    >>>     metadata_obj,
    >>>     Column("ID", Integer, primary_key=True),
    >>>     Column("FirstName", String(60), nullable=False),
    >>>     Column("LastName", String(60), nullable=False),
    >>> )
    >>> template = "http://example.com/Student/{\"ID\"}/{\"FirstName\"}-{\"LastName\"}"
    >>> from rdflib_r2r.r2r_store import ColForm, sql_pretty
    >>> colform = ColForm.from_template(students_table, template, irisafe=False)
    >>> display(colform)
    >>> print(sql_pretty( colform.expr() ))
    >>> display(ColForm.to_subforms_columns(colform))
    """

    #: booleans indicate whether to SQL-escape the columns
    form: Tuple[Union[str, bool, NoneType],...]
    cols: Tuple[ColumnElement,...]

    def __init__(self, form:Iterable[str|bool|NoneType], cols:Iterable[ColumnElement]):
        self.form, self.cols = tuple(form), tuple(cols)

    def __hash__(self):
        return hash((self.form, self.cols))

    def __repr__(self):
        return f"ColForm(form={self.form}, cols={self.cols})"

    def expr(self) -> ColumnElement:
        """Turn this ColForm into a SQL expression object"""
        if self.cols == ():
            return literal_column("".join(cast(tuple[str],self.form)))
        if list(self.form) == [None]:
            return self.cols[0]
        cols = list(self.cols)
        parts = []
        for formpart in self.form:
            if formpart in [True, None, False]:
                col = cols.pop(0)
                if col.type != sqltypes.VARCHAR:
                    col = sqlfunc.cast(col, sqltypes.VARCHAR)
                # non-string values indicate whether to escape URI terms
                part = sql_safe(col) if (formpart == True) else col
            else:
                part = formpart
                # part = sqlfunc.cast(literal(formpart), sqltypes.VARCHAR)
            parts.append(part)
        return functools.reduce(operator.add, parts)

    @classmethod
    def from_template(cls, dbtable, template, irisafe=False) -> "ColForm":
        """Parse RDB2RDF template into ColForm object"""
        # make python format string: escape curly braces by doubling {{ }}
        template = re.sub("\\\\[{}]", lambda x: x.group(0)[1] * 2, template)

        form, cols = [], []
        for prefix, colname, _, _ in Formatter().parse(template):
            if prefix != "":
                form.append(prefix)
            if colname:
                col = R2RStore._get_col(dbtable, colname, template=True)
                form.append(irisafe)
                cols.append(col)

        return cls(form, cols)

    @classmethod
    def from_subform(cls, cols, idxs, form) -> "ColForm":
        """A subform is a tuple of (external indexes, form sequence of strings) """
        cols = list(cols)
        return cls(form, [cols[i] for i in idxs])

    @classmethod
    def from_expr(cls, expr:ColumnElement) -> "ColForm":
        return cls([None], [expr])

    @classmethod
    def null(cls) -> "ColForm":
        return cls.from_expr(null())

    @staticmethod
    def to_subforms_columns(*colforms:"ColForm") -> Tuple[List[SubForm], List[ColumnElement]]:
        """Return a tuple of subforms and columns for these colforms"""
        subforms:List[SubForm] = [] 
        allcols:List[ColumnElement] = []
        i = 0
        for cf in colforms:
            cols = [c for c in cf.cols if type(c) != str]
            if cf.form:
                subforms.append(SubForm(range(i, i + len(cols)), cf.form))
                i += len(cols)
                allcols += list(cols)
        return subforms, allcols

    @staticmethod
    def equal(*colforms, eq=True) -> Generator[ColumnElement,None,None]:
        if colforms:
            # Sort colforms by descending frequency of form (for efficient equalities)
            form_count = Counter(cf.form for cf in colforms)
            colforms = sorted(colforms, key=lambda cf: -form_count[cf.form])

            cf0, *cfs = colforms
            expr0 = cf0.expr()
            for cf in cfs:
                if tuple(cf0.form) == tuple(cf.form):
                    for c0, c in zip(cf0.cols, cf.cols):
                        yield (c0 == c) if eq else (c0 != c)
                else:
                    logging.warn(f"Cannot reduce {cf0} and {cf}")
                    # TODO: fancy prefix checking
                    yield (expr0 == cf.expr()) if eq else (expr0 != cf.expr())

    @classmethod
    def op(cls, opstr:str, cf1:"ColForm", cf2:"ColForm") -> "ColForm":
        if opstr in ["=", "=="]:
            return cls.from_expr(sql_and(*cls.equal(cf1, cf2)))
        elif opstr in ["!=", "<>"]:
            return cls.from_expr(sql_or(*cls.equal(cf1, cf2, eq=False)))
        elif opstr == "/":
            op = sqlalchemy.sql.operators.custom_op(opstr, is_comparison=True)
            a, b = cf1.expr(), cf2.expr()
            r = sqlfunc.cast(op(a, b), sqltypes.REAL)
            return cls.from_expr(r)
        else:
            op = sqlalchemy.sql.operators.custom_op(opstr, is_comparison=True)
            # TODO: fancy type casting
            return cls.from_expr(op(cf1.expr(), cf2.expr()))



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


class R2RStore(Store):
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

    @classmethod
    def _term_map_colforms(
        cls, graph: Graph, dbtable: Table | NamedFromClause | Subquery, parent: Node, wheres: List[ColumnElement[bool]], 
            mapper: Node, shortcut:Node, obj=False
    ) -> Generator[tuple[ColForm, Table | NamedFromClause | Subquery], Any, Any]: 
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
                yield ColForm([f"'{const.n3()}'"], []), dbtable
        elif graph.value(parent, mapper):
            for tmap in graph.objects(parent,mapper):
                if graph.value(tmap, rr.constant):
                    # constant value
                    for const in graph.objects(tmap, rr.constant):
                        yield ColForm([f"'{const.n3()}'"], []), dbtable
                else:
                    termtype = graph.value(tmap, rr.termType) or rr.IRI
                    if graph.value(tmap, rr.column):
                        colname = graph.value(tmap, rr.column)
                        assert colname
                        colform = ColForm.from_expr(cls._get_col(dbtable, str(colname)))
                        if obj:
                            # for objects, the default term type is Literal
                            termtype = graph.value(tmap, rr.termType) or rr.Literal
                    elif graph.value(tmap, rr.template):
                        template = graph.value(tmap, rr.template)
                        colform = ColForm.from_template(
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
                            colform = ColForm(colform.form, cols)
                            yield colform, table
                        continue
                    else:
                        # TODO: replace with RDB-specific construct (postgresql?)
                        rowid = literal_column(f'"{dbtable.name}".rowid').cast(
                            sqltypes.VARCHAR
                        )
                        form = ["_:" + dbtable.name.replace('_ref','') + "#", None]
                        yield ColForm(form, [rowid]), dbtable
                        continue

                    if termtype == rr.IRI:
                        form = ["<"] + list(colform.form) + [">"]
                        yield ColForm(form, colform.cols), dbtable
                    elif termtype == rr.BlankNode:
                        yield ColForm((["_:"] + list(colform.form)), colform.cols), dbtable
                    elif obj:
                        if graph.value(tmap, rr.language):
                            lang = graph.value(tmap, rr.language)
                            cols = [
                                sqlfunc.cast(c, sqltypes.VARCHAR) for c in colform.cols
                            ]
                            form = ['"'] + list(colform.form) + ['"@' + str(lang)]
                            yield ColForm(form, cols), dbtable
                        elif graph.value(tmap, rr.datatype):
                            dtype = graph.value(tmap, rr.datatype)
                            assert isinstance(dtype, URIRef)
                            cols = [
                                sqlfunc.cast(c, sqltypes.VARCHAR) for c in colform.cols
                            ]
                            form = ['"'] + list(colform.form) + ['"^^' + dtype.n3()]
                            yield ColForm(form, cols), dbtable
                        else:
                            # keep original datatype
                            yield colform, dbtable
                    else:
                        # not a real literal
                        yield ColForm.from_expr(literal_column("'_:'")), dbtable

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
        ) or [(ColForm.null(), dbtable)]

        # Class Map
        if (not pfilt) or (None in pfilt) or (RDF.type == qp):
            for c in mg.objects(s_map, rr["class"]):
                pcolform = ColForm([f"'{RDF.type.n3()}'"], [])
                ocolform = ColForm([f"'{c.n3()}'"], [])
                # no unsafe IRI because it should be defined to be safe
                if (qo is not None) and (qo != c):
                    continue
                for gcolform, gtable in gcolforms:
                    subforms, cols = ColForm.to_subforms_columns(
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
            ) or [(ColForm.null(), dbtable)]
            for pcolform, ptable in pcolforms:
                pstr = "".join([ str(fe) for fe in pcolform.form if fe ])
                if (qp is not None) and pstr[1:-1] != qp.n3():
                    # Filter out non-identical property patterns
                    continue
                for ocolform, otable in ocolforms:
                    for gcolform, gtable in gcolforms:
                        where = swhere + pwhere + owhere
                        subforms, cols = ColForm.to_subforms_columns(
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
                col = ColForm.from_subform(cols, *subform).expr()
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
                col = ColForm.from_subform(cols, *subform).expr()
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
                    (q, ColForm.from_subform(cols, *subform))
                    for q, subform in zip((qs, qp, qo), subforms)
                    if isinstance(q, Variable)
                ]
                if len(pat_query._from_obj) == 1:
                    # Single table, so try to merge shared-subject terms
                    table = pat_query._from_obj[0], pat_query.whereclause
                    table_varcolforms.setdefault(table, set()).update(qvar_colform)
                else:
                    qvars, colforms = zip(*qvar_colform)
                    subforms, allcols = ColForm.to_subforms_columns(*colforms)
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
            subform, allcols = ColForm.to_subforms_columns(*colforms)
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
                var_colforms.setdefault(var, []).append(ColForm(form, cols))

        # Simplify colform equalities
        colforms = [cfs[0] for cfs in var_colforms.values()]
        subforms, allcols = ColForm.to_subforms_columns(*colforms)
        where = [eq for cs in var_colforms.values() for eq in ColForm.equal(*cs)]
        return SelectVarSubForm(select(*allcols).where(*where), dict(zip(var_colforms, subforms)))

    def queryExpr(self, conn: Connection, expr, var_cf:dict[Variable, ColForm]) -> ColForm:
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
                return ColForm.from_expr(func(sub.cols[0]))
            else:
                return ColForm.from_expr(func(sub.expr()))

        if hasattr(expr, "name") and (expr.name == "RelationalExpression"):
            a = self.queryExpr(conn, expr.expr, var_cf)
            b = self.queryExpr(conn, expr.other, var_cf)
            return ColForm.op(expr.op, a, b)

        math_expr_names = ["MultiplicativeExpression", "AdditiveExpression"]
        if hasattr(expr, "name") and (expr.name in math_expr_names):
            # TODO: ternary ops?
            a = self.queryExpr(conn, expr.expr, var_cf)
            for other in expr.other:
                b = self.queryExpr(conn, other, var_cf)
                return ColForm.op(expr.op[0], a, b)

        if hasattr(expr, "name") and (expr.name == "ConditionalAndExpression"):
            exprs = [self.queryExpr(conn, e, var_cf) for e in [expr.expr] + expr.other]
            return ColForm.from_expr(sql_and(*[e.expr() for e in exprs]))

        if hasattr(expr, "name") and (expr.name == "ConditionalOrExpression"):
            exprs = [self.queryExpr(conn, e, var_cf) for e in [expr.expr] + expr.other]
            return ColForm.from_expr(sql_or(*[e.expr() for e in exprs]))

        if hasattr(expr, "name") and (expr.name == "Function"):
            # TODO: it would be super cool to do UDFs here
            if expr.iri in XSDToSQL:
                for e in expr.expr:
                    cf = self.queryExpr(conn, e, var_cf)
                    val = sqlfunc.cast(cf.expr(), XSDToSQL[expr.iri])
                    return ColForm.from_expr(val)

        if hasattr(expr, "name") and (expr.name == "UnaryNot"):
            cf = self.queryExpr(conn, expr.expr, var_cf)
            return ColForm.from_expr( sqlalchemy.not_(cf.expr()) ) 
        
        if hasattr(expr, "name") and (expr.name == "Builtin_BOUND"):
            cf = self.queryExpr(conn, expr.arg, var_cf)
            return ColForm.from_expr( cf.expr().is_(None) ) 

        if isinstance(expr, str) and (expr in var_cf):
            return var_cf[expr]
        if isinstance(expr, URIRef):
            return ColForm.from_expr(expr.n3())
        if isinstance(expr, Literal):
            return ColForm.from_expr(expr.toPython())
        if isinstance(expr, str):
            return ColForm.from_expr(toPython(cast(Node,from_n3(expr))))

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
                var_colforms.setdefault(v, []).append(ColForm.from_subform(cols1, *sf1))
            cols2 = list(query2.exported_columns)
            for v, sf2 in var_subform2.items():
                var_colforms.setdefault(v, []).append(ColForm.from_subform(cols2, *sf2))

            where = [eq for cs in var_colforms.values() for eq in ColForm.equal(*cs)]
            return SelectVarSubForm(part_query.filter(~query2.where(*where).exists()), var_subform)

        cols = list(getattr(part_query, "exported_columns", part_query.c))
        var_cf = {v: ColForm.from_subform(cols, *sf) for v, sf in var_subform.items()}
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
            var_colforms.setdefault(v, []).append(ColForm.from_subform(cols1, *sf1))
        cols2 = list(query2.c)
        for v, sf2 in var_subform2.items():
            var_colforms.setdefault(v, []).append(ColForm.from_subform(cols2, *sf2))

        colforms = [cfs[0] for cfs in var_colforms.values()]
        subforms, allcols = ColForm.to_subforms_columns(*colforms)
        where = [eq for cs in var_colforms.values() for eq in ColForm.equal(*cs)]
        return SelectVarSubForm(select(*allcols).where(*where), dict(zip(var_colforms, subforms)))

    def queryAggregateJoin(self, conn: Connection, agg) -> SelectVarSubForm:
        # Assume agg.p is always a Group
        group_expr, group_part = agg.p.expr, agg.p.p
        part_query, var_subform = self.queryPart(conn, group_part).as_tuple()
        cols = part_query.c
        var_cf = {v: ColForm.from_subform(cols, *sf) for v, sf in var_subform.items()}

        # Get aggregate column expressions
        var_agg = {a.res: self.queryExpr(conn, a, var_cf) for a in agg.A}
        groups = [
            c
            for e in (group_expr or [])
            for c in self.queryExpr(conn, e, var_cf).cols
            if type(c) != str
        ]

        subforms, allcols = ColForm.to_subforms_columns(*var_agg.values())
        query = select(*allcols).group_by(*groups)
        return SelectVarSubForm(query, dict(zip(var_agg, subforms)))

    def queryExtend(self, conn: Connection, part) -> SelectVarSubForm:
        part_query, var_subform = self.queryPart(conn, part.p).as_tuple()
        assert isinstance(part_query, Select)  # ?
        cols = list(part_query.exported_columns)
        var_cf = {v: ColForm.from_subform(cols, *sf) for v, sf in var_subform.items()}

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
        colforms = [ColForm.from_subform(cols, *sf) for sf in var_subform.values()]
        subforms, allcols = ColForm.to_subforms_columns(*colforms)
        part_query = part_query.with_only_columns(*allcols)
        return SelectVarSubForm(part_query, dict(zip(var_subform, subforms)))

    def queryOrderBy(self, conn: Connection, part) -> SelectVarSubForm:
        part_query, var_subform = self.queryPart(conn, part.p).as_tuple()
        cols = list(part_query.exported_columns)
        var_cf = {v: ColForm.from_subform(cols, *sf) for v, sf in var_subform.items()}

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
                e1 = ColForm.from_subform(cols1, *var_subform1[v]).expr()
            else:
                e1 = null()
            allcols1.append(e1.label(str(v)))
            if v in var_subform2:
                e2 = ColForm.from_subform(cols2, *var_subform2[v]).expr()
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

        var_colforms:dict[Variable,List[ColForm]] = {}
        allcols1, allcols2 = [], []
        cols1 = list(query1.c)
        for v, sf1 in var_subform1.items():
            cf = ColForm.from_subform(cols1, *sf1)
            var_colforms.setdefault(v, []).append(cf)
            allcols1.append(cf.expr().label(str(v)))
        
        query2 = query2.subquery()
        cols2 = list(query2.c)
        for v, sf2 in var_subform2.items():
            cf = ColForm.from_subform(cols2, *sf2)
            var_colforms.setdefault(v, []).append(cf)
            allcols2.append(cf.expr().label(str(v)))

        where = [eq for cs in var_colforms.values() for eq in ColForm.equal(*cs)]

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
                ColForm.from_subform(query.exported_columns, *sf).expr().label(str(var))
                for var, sf in var_subform.items()
            ]
            return query.with_only_columns(*cols)
        else:
            cols = [
                ColForm.from_subform(query.c, *sf).expr().label(str(var))
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
