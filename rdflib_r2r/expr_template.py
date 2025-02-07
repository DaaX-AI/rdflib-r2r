from rdflib_r2r.r2r_store import R2RStore, SubForm, sql_safe


import sqlalchemy.sql.operators
from sqlalchemy import and_ as sql_and, func as sqlfunc, literal_column, null, or_ as sql_or, types as sqltypes
from sqlalchemy.sql.expression import ColumnElement


import functools
import logging
import operator
import re
from collections import Counter
from string import Formatter
from types import NoneType
from typing import Generator, Iterable, List, Tuple, Union, cast



class ExpressionTemplate:
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
        ci = 0
        parts = []
        for formpart in self.form:
            if formpart in [True, None, False]:
                col = self.cols[ci]
                if col.type != sqltypes.VARCHAR:
                    col = sqlfunc.cast(col, sqltypes.VARCHAR)
                # non-string values indicate whether to escape URI terms
                part = sql_safe(col) if (formpart == True) else col
                ci += 1
            else:
                part = formpart
                # part = sqlfunc.cast(literal(formpart), sqltypes.VARCHAR)
            parts.append(part)
        return functools.reduce(operator.add, parts)

    @classmethod
    def from_template(cls, dbtable, template, irisafe=False) -> "ExpressionTemplate":
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
    def from_subform(cls, cols, idxs, form) -> "ExpressionTemplate":
        """A subform is a tuple of (external indexes, form sequence of strings) """
        cols = list(cols)
        return cls(form, [cols[i] for i in idxs])

    @classmethod
    def from_expr(cls, expr:ColumnElement) -> "ExpressionTemplate":
        return cls([None], [expr])

    @classmethod
    def null(cls) -> "ExpressionTemplate":
        return cls.from_expr(null())

    @staticmethod
    def to_subforms_columns(*colforms:"ExpressionTemplate") -> Tuple[List[SubForm], List[ColumnElement]]:
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
    def op(cls, opstr:str, cf1:"ExpressionTemplate", cf2:"ExpressionTemplate") -> "ExpressionTemplate":
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