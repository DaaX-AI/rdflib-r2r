import logging
import time
from typing import List, Tuple
import pandas
import pyodbc
import sqlalchemy
from rdflib import Graph

from rdflib_r2r.new_r2r_store import NewR2rStore
from rdflib_r2r.r2r_store import SQL_FUNC

def generate_sql(df:pandas.DataFrame, db:sqlalchemy.Engine, mapping_file:str):
    source_sparql = df.loc[:,'sparql'].fillna('').tolist()
    results:List[Tuple[str,str,str]] = []

    mg = Graph().parse(mapping_file)
    store = NewR2rStore(db, mg)

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

def calculate_timings(connect_str:str, dbpass:str, df:pandas.DataFrame, results:List[Tuple[int,float,int|None,str|None,str|None]], 
                      field="sql2") -> List[Tuple[int,float,int|None,str|None,str|None]]:
    """Loads the SQL queries and runs them, noting how long they took and how many results they returned."""

    done_ids = { r[0] for r in results }

    def connect() -> pyodbc.Connection:
        cxn = pyodbc.connect(connect_str, password=dbpass, timeout=300)
        cxn.timeout = 300 # The value passes to connect above seems to be ignored?
        return cxn

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
            assert isinstance(id,int)
            if id in done_ids:
                continue

            if not sql or not isinstance(sql, str):
                results.append((id, float("nan"), None, None, None))
                continue

            result_count=None
            err=None
            t0 = time.time()
            try:
                qrs = csr.execute(sql).fetchmany(5000)
                t1 = time.time()
                result_count = len(qrs)
                
                print(f'Q{id}: {t1-t0}, {result_count} results, first {qrs[0]}')
                results.append((id, t1-t0, result_count, err, str(qrs[0])))
                done_ids.add(id)
            except KeyboardInterrupt:
                break
            except pyodbc.OperationalError:
                recover()
            except Exception as e:
                t1 = time.time()
                err=str(e)
                results.append((id, t1-t0, result_count, err, None))
                done_ids.add(id)
            finally:
                try:
                    csr.cancel()
                except:
                    recover()

            rlast = results[-1]
            logging.info(f"Q{rlast[0]}: {rlast[1]}, {rlast[2]} results, error: {rlast[3] or 'None'}, first: {rlast[4] or 'None'}")

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
