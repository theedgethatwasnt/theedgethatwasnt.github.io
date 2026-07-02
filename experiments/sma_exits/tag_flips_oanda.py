import os, v20, duckdb
ctx=v20.Context('api-fxtrade.oanda.com','443',token=os.environ['OANDA_API_KEY'])
acct=os.environ['OANDA_ACCOUNT_ID_010']
STACK={'EUR_JPY','EUR_USD','USD_JPY','GBP_USD','GBP_JPY'}
# find OANDA trade IDs whose close reason is MARKET_ORDER (the confirmed flips)
flip_ids=set()
for t in ctx.trade.list(acct, state='CLOSED', count=500).body.get('trades',[]):
    if t.instrument not in STACK or t.openTime[:10]<'2026-06-02': continue
    for cid in (t.closingTransactionIDs or []):
        tx=ctx.transaction.get(acct,cid).body.get('transaction')
        if getattr(tx,'reason','')=='MARKET_ORDER':
            flip_ids.add(str(t.id)); break
con=duckdb.connect('/data/db/trades.duckdb')
n=0
for oid in flip_ids:
    r=con.execute("UPDATE trades SET exit_reason='flip' WHERE trade_id=? AND account_id='<OANDA_ACCOUNT_ID>'",[f'010_{oid}'])
    n+=1
tot=con.execute("SELECT count(*), round(sum(pnl_pips),1) FROM trades WHERE account_id='<OANDA_ACCOUNT_ID>' AND exit_reason='flip'").fetchone()
con.close()
print(f"tagged {len(flip_ids)} OANDA flip trades; DB now has {tot[0]} 'flip' records totalling {tot[1]}p")
