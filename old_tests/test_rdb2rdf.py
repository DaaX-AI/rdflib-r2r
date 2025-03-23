"""
Run tests from [https://www.w3.org/TR/2012/NOTE-rdb2rdf-test-cases-20120814/]

Downloaded from [https://dvcs.w3.org/hg/rdb2rdf-tests/raw-file/default/rdb2rdf-ts.zip]
"""

import pathlib
import logging
import re
import pathlib
from typing import NamedTuple
import sys, os


import pytest
import rdflib
from rdflib.namespace import RDF, Namespace
from rdflib.compare import to_isomorphic, graph_diff
from rdflib.util import from_n3

sys.path.append( os.path.dirname(os.path.realpath(__file__)) )
from rdflib_r2r.r2r_mapping import mapping_from_db
from util import setup_engine, create_database
from rdflib_r2r import R2RStore

test = Namespace("http://www.w3.org/2006/03/test-description#")
dcterms = Namespace("http://purl.org/dc/elements/1.1/")
rdb2rdftest = Namespace("http://purl.org/NET/rdb2rdf-test#")
base = Namespace("http://www.w3.org/2001/sw/rdb2rdf/test-cases/#")

PATH = pathlib.Path(__file__).parent / "rdb2rdf-ts"
if not PATH.exists():
    import zipfile
    fzip = pathlib.Path(__file__).parent / "rdb2rdf-ts.zip"
    with zipfile.ZipFile(fzip, 'r') as zip_ref:
        zip_ref.extractall(fzip.parent)

class TestCase(NamedTuple):
    id: str
    sql_fname: str
    path: pathlib.Path
    meta: dict


TestCase.__test__ = False


def yield_database_testcases(path: pathlib.Path):
    manifest_file = str(path.joinpath("manifest.ttl"))

    g = rdflib.Graph()
    g.parse(manifest_file, format="ttl")
    # Get test databases
    for dbspec in g[: RDF.type : rdb2rdftest.DataBase]:
        logging.warn("Loading w3c rdb2rdf DB %s ('%s')", path.name, g.value(dbspec, dcterms.title))
        sql_fname = g.value(dbspec, rdb2rdftest.sqlScriptFile)

        for testspec in g.objects(dbspec, rdb2rdftest.relatedTestCase):
            id = g.value(testspec, dcterms.identifier)
            if id:
                marks = []
                if not g.value(testspec, rdb2rdftest.hasExpectedOutput).toPython():
                    marks.append(pytest.mark.xfail)
                yield pytest.param(
                    TestCase(
                        id=str(id),
                        sql_fname=sql_fname,
                        path=path,
                        meta=dict(g[testspec:]),
                    ),
                    id=str(id),
                    marks=marks,
                )


PATHS = sorted(
    path
    for path in PATH.iterdir()
    if path.name[0] != "."
)

TESTS = [
    testcase
    for path in PATHS
    for testcase in sorted(yield_database_testcases(path), key=lambda s: s.id)
]

@pytest.fixture()
def dbecho(pytestconfig):
    return pytestconfig.getoption("dbecho")

@pytest.fixture()
def nopattern(pytestconfig):
    return pytestconfig.getoption("nopattern")

@pytest.mark.parametrize("testcase", TESTS, ids=[t.id for t in TESTS])
@pytest.mark.parametrize("engine_name", ["sqlite", "duckdb"])
def test_rdb2rdf(testcase: TestCase, engine_name: str, dbecho: bool, nopattern: bool):
    test_out = pathlib.Path(f'docs/test-results/{engine_name}-rdb2rdf/')
    test_out.mkdir(parents=True, exist_ok=True)
    test_file = test_out.joinpath(f"{testcase.id}.md")
    report = f"# {testcase.id}\n"
    report += f"[link](https://www.w3.org/TR/rdb2rdf-test-cases/#{testcase.id})\n"
    report += testcase.meta.get(dcterms.title) + "\n\n"

    try:
        # Create database
        db = setup_engine(engine_name, echo=dbecho)
        create_database(db, testcase.path, testcase.sql_fname)

        # Create mapped R2RStore
        mapping = None
        if rdb2rdftest.mappingDocument in testcase.meta:
            # Load mapping
            mapfile = testcase.path.joinpath(testcase.meta[rdb2rdftest.mappingDocument])
            fmt = rdflib.util.guess_format(str(mapfile))
            mapping_graph = rdflib.Graph().parse(str(mapfile), format=fmt)
        else:
            # Make direct mapping
            mapping_graph = mapping_from_db(db)

        g_made = rdflib.Graph(R2RStore(db=db, mapping_graph=mapping_graph))

        if not rdb2rdftest.output in testcase.meta:
            tuple(g_made)
            return

        outfile = testcase.path.joinpath(testcase.meta[rdb2rdftest.output])
        fmt = rdflib.util.guess_format(str(outfile))
        if fmt == "nquads":
            # RDFLib nquads parser doesn't work?
            rx = re.compile(r"([^ ]+) ([^ ]+) (.+?) ?(?: (<[^>]+>))? \.$")
            g_goal = rdflib.ConjunctiveGraph()
            for li, l in enumerate(open(outfile).read().splitlines()):
                if l.strip():
                    m = rx.match(l.strip())
                    if m:
                        g_goal.add(tuple(from_n3(n) for n in m.groups()))
                    else:
                        logging.warn(f"bad nquads line {li} in {outfile}: {l}")
            logging.warn(("g_goal", type(g_goal), len(list(g_goal)), list(g_goal)))
        else:
            g_goal = rdflib.Graph().parse(str(outfile), format=fmt)
            logging.warn(
                (
                    "g_goal",
                    type(g_goal),
                    len(list(g_goal)),
                    g_goal.serialize(format="turtle"),
                )
            )

        

        iso_made, iso_goal = to_isomorphic(g_made), to_isomorphic(g_goal)
        in_both, in_made, in_goal = graph_diff(iso_made, iso_goal)
        
        def dump_nt_sorted(g):
            return sorted(g.serialize(format="nt").strip().splitlines())

        diff_lines = []
        for g,p in zip([in_both, in_made, in_goal], ['','+ ','- ']):
            for line in dump_nt_sorted(g):
                diff_lines.append((line, p))
        difftxt = '\n'.join(p+line for line, p in sorted(diff_lines))
        
        sql_query = g_made.store.getSQL("SELECT * WHERE {?s ?p ?o}")
        report += f"## Created SQL query\n```sql\n{sql_query}\n```\n\n"

        dump_made = '\n'.join(dump_nt_sorted(g_made))
        logging.warn(f"Raw triples:\n{dump_made}")
        report += f"## Raw ouput triples\n```\n{dump_made}\n```\n\n"
        
        report += "## Triple Diff\n```diff\n" + difftxt + "\n```\n\n"
        report += 'Full query: ' + ('SUCCES' if iso_made == iso_goal else 'FAIL')
        test_file.write_text(report)

        for li, line in enumerate(dump_nt_sorted(in_both)):
            logging.warn(f"in_both {li+1}/{len(list(in_both))}: {line}")
        for li, line in enumerate(dump_nt_sorted(in_made)):
            logging.warn(f"in_made {li+1}/{len(list(in_made))}: {line}")
        for li, line in enumerate(dump_nt_sorted(in_goal)):
            logging.warn(f"in_goal {li+1}/{len(list(in_goal))}: {line}")
        assert iso_made == iso_goal

        if not nopattern:
            report += '## Pattern tests\n'
            # Test graph pattern filters
            made_triples = sorted(g_made)
            if any(made_triples):
                g_ss, g_ps, g_os = zip(*made_triples)
                report += '### Subject Pattern tests\n'
                for s in sorted(set(g_ss)):
                    if isinstance(s, rdflib.term.BNode):
                        continue
                    s_triples = sorted(g_made.triples([s, None, None]))
                    assert s_triples, f'subject filter {s}'
                    for s_, _, _ in s_triples:
                        assert s_ == s, 'subject equality'

                report += '### Predicate Pattern tests\n'
                for p in sorted(set(g_ps)):
                    if isinstance(p, rdflib.term.BNode):
                        continue
                    p_triples = sorted(g_made.triples([None, p, None]))
                    assert p_triples, f'predicate filter {p}'
                    for _, p_, _ in p_triples:
                        assert p_ == p, 'predicate equality'

                report += '### Object Pattern tests\n'
                for o in sorted(set(g_os)):
                    if isinstance(o, rdflib.term.BNode):
                        continue
                    o_triples = sorted(g_made.triples([None, None, o]))
                    assert o_triples, f'object filter {o}'
                    for _, _, o_ in o_triples:
                        assert o_.toPython() == o.toPython(), 'object equality'
            
    except Exception as e:
        import traceback, os
        tb = traceback.format_exc().replace(os.getcwd(), '')
        report += f"\n\n```\n{tb}\n```"
        test_file.write_text(report)
        raise e
        

def test_synthesis(module_results_df):
    df = module_results_df
    ids = df.testcase.apply(lambda x: x.id)
    df["link"] = (
        "[" + ids + "]" + "(https://www.w3.org/TR/rdb2rdf-test-cases/#" + ids + ")"
    )
    df["title"] = df.testcase.apply(lambda x: x.meta.get(dcterms.title))
    status_emoji = {
        "passed": "✅",
        "failed": "❌",
        "xfailed": "🔆",
        "xpassed": "❗️",
    }
    logging.warn(("statuses", set(df["status"])))

    testdir = pathlib.Path('docs/test-results/')
    testdir.mkdir(parents=True, exist_ok=True)

    def get_status_link(row):
        logging.warn(row)
        testcase = row.testcase
        text = status_emoji.get(row.status, "") + " " + row.status
        return f"[{text}]({row.engine_name}-rdb2rdf/{testcase.id}.md)"

    df["status"] = df.apply(get_status_link, axis=1)

    with testdir.joinpath("rdb2rdf.md").open("w") as fw:
        print("# RDB2RDF Test results\n", file=fw)
        df = df[["engine_name", "link", "status", "duration_ms", "title"]]
        print(df.to_markdown(index=False), file=fw)
