import os, v20, duckdb
PIPS={'EUR_JPY':0.01,'USD_JPY':0.01,'EUR_USD':0.0001,'GBP_USD':0.0001,'GBP_JPY':0.01}
ctx=v20.Context('api-fxtrade.oanda.com','443',token=os.environ['OANDA_API_KEY'])
acct=os.environ['OANDA_ACCOUNT_ID_010']
con=duckdb.connect('/data/db/trades.duckdb')
rows=con.execute("""SELECT trade_id, pair, pnl_pips, exit_reason FROM trades
  WHERE account_id='<OANDA_ACCOUNT_ID>' AND label LIKE 'sma_stack%'
  AND exit_time IS NOT NULL AND trade_id ~ '^010_[0-9]+$' ORDER BY entry_time""").fetchall()
print(f"re-deriving {len(rows)} matchable 010 sma_stack records from OANDA ground truth:")
before=con.execute("SELECT round(sum(pnl_pips),1) FROM trades WHERE account_id='<OANDA_ACCOUNT_ID>' AND label LIKE 'sma_stack%' AND exit_time IS NOT NULL").fetchone()[0]
fixed=0; flips=0
for trade_id,pair,old_pnl,old_reason in rows:
    oid=trade_id.split('_')[1]; pip=PIPS.get(pair,0.0001)
    try:
        t=ctx.trade.get(acct,oid).body.get('trade')
        if t is None or getattr(t,'averageClosePrice',None) is None: continue
        d=1 if float(t.initialUnits)>0 else -1
        pnl=round((float(t.averageClosePrice)-float(t.price))/pip*d,1)
        con.execute("UPDATE trades SET exit_price=?, pnl_pips=?, exit_reason='oanda_truth' WHERE trade_id=?",
                    [float(t.averageClosePrice), pnl, trade_id])
        fixed+=1
        if (old_pnl>0) != (pnl>0): flips+=1
    except Exception as e:
        print(f"  {trade_id}: {e}")
after=con.execute("SELECT round(sum(pnl_pips),1) FROM trades WHERE account_id='<OANDA_ACCOUNT_ID>' AND label LIKE 'sma_stack%' AND exit_time IS NOT NULL").fetchone()[0]
con.close()
print(f"  fixed {fixed} records, {flips} flipped win<->loss sign")
print(f"  010 sma_stack DB total: {before}p -> {after}p")
