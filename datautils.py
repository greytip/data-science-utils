
import json
import numpy as np
import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import session

import settings
verticals = None

def preprocess(df):
    ######################## DATA Pre-Processing and Cleanups######
    common_drop_columns = ['audit_timestamp', 'audit_source', 'id']
    df.drop(common_drop_columns,1, inplace=True, errors='ignore')
    return df

def dataload(conn_or_session, table_name, preprocess_func=preprocess, batch=False,
             sample=False, sample_pct=5, sample_type='BERNOULLI'):
    global verticals
    if os.path.exists('../greytip_stuff/verticals.json'):
        verticals = json.load(open('../greytip_stuff/verticals.json'))
    else:
        verticals = []

    if os.path.exists('../greytip_stuff/' + table_name + '.csv'):
        return preprocess_func(pd.read_csv('../greytip_stuff/' + table_name + '.csv',
                          encoding='utf-8-sig',  sep=','))
    else:
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

            return preprocess_func(pd.read_s)
        # tables_df[table_name].to_csv('../greytip_stuff/' + table_name + '.csv', index=False, sep=',',
        #                                encoding='utf-8')

    for key in rev_verticals:
        tables_df['v_account_summary'].replace(key, rev_verticals[key],inplace=True)

    return tables_df


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
