import os, v20, duckdb
PIP=0.0001
ctx=v20.Context('api-fxtrade.oanda.com','443',token=os.environ['OANDA_API_KEY'])
acct=os.environ['OANDA_ACCOUNT_ID_010']
con=duckdb.connect('/data/db/trades.duckdb')  # read-write
rows=con.execute("""SELECT trade_id, entry_price, direction FROM trades
  WHERE account_id='<OANDA_ACCOUNT_ID>' AND label='sma_stack_GBP_USD'
  AND exit_reason='closed_unknown' AND trade_id ~ '^010_[0-9]+$' ORDER BY entry_time""").fetchall()
print(f"backfilling {len(rows)} GBP_USD trades with real OANDA close fills:")
fixed=0; total_recovered=0.0
for trade_id, entry_px, direction in rows:
    oid=trade_id.split('_')[1]
    try:
        r=ctx.trade.get(acct, oid)
        t=r.body.get('trade')
        if t is None or getattr(t,'averageClosePrice',None) is None:
            print(f"  {trade_id}: no close price from OANDA (state={getattr(t,'state','?')})"); continue
        close_px=float(t.averageClosePrice); units=float(t.initialUnits)
        rpl=float(getattr(t,'realizedPL',0))
        d=1 if units>0 else -1
        pnl_pips=(close_px-entry_px)/PIP*d
        con.execute("""UPDATE trades SET exit_price=?, pnl_pips=?, exit_reason='psar_backfill'
            WHERE trade_id=?""",[close_px, round(pnl_pips,1), trade_id])
        total_recovered+=pnl_pips; fixed+=1
        print(f"  {trade_id}: entry={entry_px:.5f} close={close_px:.5f} -> {pnl_pips:+.1f}p (rPL=${rpl:+.4f})")
    except Exception as e:
        print(f"  {trade_id}: ERROR {e}")
con.close()
print(f"\nfixed {fixed} trades, recovered {total_recovered:+.1f}p (was recorded as ~0)")
