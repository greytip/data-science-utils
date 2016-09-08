
import json
import numpy as np
import os
import pandas as pd
from sqlalchemy import create_engine

import settings
verticals = None

def preprocess(df):
    ######################## DATA Pre-Processing and Cleanups######
    common_drop_columns = ['audit_timestamp', 'audit_source', 'id']
    df.drop(common_drop_columns,1, inplace=True, errors='ignore')
    return df

def dataload(conn_or_session, table_name=None, custom_sql=None, preprocess_func=preprocess,
             batch=False, batch_size=None, sample=False, sample_pct=5, sample_type='BERNOULLI'):

    from sqlalchemy.orm import session
    assert (bool(table_name) ^ bool(custom_sql)), "Only table_name or custom_sql allowed"
    #TODO: implement  sampling from csv also
    if custom_sql:
        fname = custom_sql
    else:
        fname = table_name
    if os.path.exists('../greytip_stuff/' + fname + '.csv'):
        return preprocess_func(pd.read_csv('../greytip_stuff/' + fname + '.csv',
                          encoding='utf-8-sig',  sep=','))
    else:
        if custom_sql:
            assert (not sample)
            return  preprocess_func(pd.read_sql(custom_sql, conn_or_session))
        if sample:
            assert sample_pct, "Sample percent mandatory"
            assert sample_type, "Sample type mandatory"
            return pd.read_sql("SELECT * from %s TABLESAMPLE %s(%d);"%(table_name,
                                                                       sample_type,
                                                                       sample_pct), conn_or_session)
        if not batch:
            return preprocess_func(pd.read_sql_table(table_name, conn_or_session, schema='dv'))
        else:
            assert isinstance(conn_or_session,session), "batch loading needs a session argument"
            assert batch_size, "batch needs  a batch_size argument"
            return preprocess_func(pd.read_sql("SELECT * FROM %s;"%(table_name),
                                                        chunksize=batch_size))

def preprocess_verticals(df):
    df = preprocess(df)
    rev_verticals = dict()
    for k,v in verticals.items():
        for val in v:
            rev_verticals.update({val:k})
    for key in rev_verticals:
        df['v_account_summary'].replace(key, rev_verticals[key],inplace=True)
    return df

if __name__ == '__main__':
    conn = create_engine(settings.local_db_url, execution_options=dict(stream_results=True))
    tables = ['apx_active_services_info']
    table_dfs = dict()
    for tablename in tables:
        table_dfs.update({tablename: dataload(conn,table_name=tablename,
                                               preprocess_func=preprocess,
                                               batch=False
                                               )})
    pass
