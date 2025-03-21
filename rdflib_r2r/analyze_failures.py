from collections import Counter
from decimal import Decimal
import logging
import re
import time
from typing import Hashable, List, Tuple
import pandas
import pyodbc
import sqlalchemy
from rdflib import Graph

from rdflib_r2r.sql_converter import SQLConverter
from rdflib_r2r.conversion_utils import SQL_FUNC

def generate_sql(df:pandas.DataFrame, db:sqlalchemy.Engine, mapping_file:str):
    source_sparql = df.loc[:,'sparql'].fillna('').tolist()
    results:List[Tuple[str,str,str]] = []

    mg = Graph().parse(mapping_file)
    store = SQLConverter(db, mg)
    logging.info(f"Generating {len(source_sparql)} SQL queries")

    for sparql in source_sparql:
        if not sparql:
            results.append(('','', 'No SPARQL'))
            continue

        try:
            sql_query = store.getSQL(sparql, initNs={'sqlf': SQL_FUNC, **{p:ns for p,ns in mg.namespaces()}})
        except Exception as e:
            results.append(('', str(e), "Failed to convert"))
            continue

        results.append((sql_query, '', "Success"))

    return pandas.concat([df[["question","sparql"]],
                         pandas.DataFrame(results, columns=['sql2', 'message', 'status'], index=df.index)],
                         axis=1)

ERROR_PREFIXES = [
    "Expr not implemented: 'Builtin_EXISTS'"
]

def display_results_overview(df:pandas.DataFrame):
    print(Counter(df["status"]))

    prefix_counts = [ len(df[(df["status"] == "Failed to convert") & df["message"].str.startswith(prefix)]) for prefix in ERROR_PREFIXES ]
    by_count = sorted(zip(ERROR_PREFIXES, prefix_counts), key=lambda p_n: -p_n[1])
    for p, n in by_count:
        if n > 0:
            print(f'{p} -> {n}')
    print()
    print(f'Total: {sum([p_n[1] for p_n in by_count])}')

    print()
    print("Others:")
    odf = df[df["status"] == "Failed to convert"]
    for p in ERROR_PREFIXES:
        odf = odf[~odf["message"].str.startswith(p)]
    print(odf["message"].head(5))    

def filter_by_message_prefix(df:pandas.DataFrame, prefix:str):
    return df[df["message"].str.startswith(prefix)]

def nice_dec(t:float) -> Decimal:
    return Decimal(t).quantize(Decimal('0.001'))

def calculate_timings(connect_str:str, dbpass:str, df:pandas.DataFrame, results:List[Tuple[Hashable,Decimal,int|None,str|None,str|None]], 
                      field="sql2") -> List[Tuple[Hashable,Decimal,int|None,str|None,str|None]]:
    """Loads the SQL queries and runs them, noting how long they took and how many results they returned."""

    done_ids = { r[0] for r in results }

    def connect() -> pyodbc.Connection:
        cxn = pyodbc.connect(connect_str, password=dbpass, timeout=300)
        cxn.timeout = 300 # The value passes to connect above seems to be ignored?
        return cxn
    
    def produce(id:Hashable, t:float, result_count:int|None, err:str|None, first:str|None):
        dt = nice_dec(t)
        logging.info(f"Q{id}: {dt}, {result_count} results, error: {err or 'None'}, first: {first or 'None'}")
        results.append((id, dt, result_count, err, first))

    conn = connect()
    conn.execute('select 1')
    csr = conn.cursor()
    try:

        def recover():
            nonlocal conn, csr
            replaced_csr = False
            replaced_conn = False

            while True:
                try:
                    if csr.execute('select 1').fetchone():
                        return
                except:
                    try:
                        csr.cancel()
                    except:
                        pass

                    if replaced_conn:
                        raise # We did all we could
                    elif replaced_csr:
                        print("Replacing connection")
                        try:
                            if conn:
                                conn.close()
                        except Exception as e:
                            logging.warning("Failed to close connection", e)                    
                        conn = None
                        conn = connect()
                        replaced_conn = True
                    else:
                        print("Replacing cursor")
                        try:
                            if csr:
                                csr.close()
                        except Exception as e:
                            logging.warning("Failed to close cursor", e)
                        csr = None
                        csr = conn.cursor()
                        replaced_csr = True
                

        for id, (sql,) in df[[field]].iterrows():
            if id in done_ids:
                continue

            if not isinstance(sql, str) or not re.sub(r'--.*','', sql).strip(): # Only comments:
                produce(id, float("nan"), None, None, None)
                continue

            result_count=None
            err=None
            t0 = time.time()
            try:
                qrs = csr.execute(sql).fetchmany(5000)
                t1 = time.time()
                result_count = len(qrs)
                
                produce(id, t1-t0, result_count, err, str(qrs[0]) if qrs else None)
                done_ids.add(id)
            except KeyboardInterrupt:
                break
            except pyodbc.OperationalError:
                recover()
            except Exception as e:
                t1 = time.time()
                err=str(e)
                produce(id, t1-t0, result_count, err, None)
                done_ids.add(id)
            finally:
                try:
                    csr.cancel()
                except:
                    recover()

        return results
    
    finally:
        try:
            if csr:
                csr.close()
        except Exception as e:
            logging.warning("Failed to close cursor", e)
        try:
            if conn:
                conn.close()
        except Exception as e:
            logging.warning("Failed to close connection", e)
