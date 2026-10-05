"""Persistent demo assignments and operator records. Never masquerades as telemetry."""
import json,sqlite3,hashlib,math
from contextlib import contextmanager
from datetime import datetime,timezone
import pandas as pd
from backend.config import DB,DATA
from backend.catalog import catalog

def now():return datetime.now(timezone.utc).isoformat()
@contextmanager
def connection():
    con=sqlite3.connect(DB,timeout=20); con.row_factory=sqlite3.Row
    try:
        with con:yield con
    finally:con.close()
def distance(a,b):
    p,q=math.radians(a['latitude']),math.radians(b['latitude']); dl=math.radians(b['longitude']-a['longitude'])
    return 6371*2*math.asin(min(1,math.sqrt(math.sin((q-p)/2)**2+math.cos(p)*math.cos(q)*math.sin(dl/2)**2)))

def initialize():
    with connection() as c:
        c.execute('CREATE TABLE IF NOT EXISTS machines (id TEXT PRIMARY KEY, mine_id TEXT, latitude REAL, longitude REAL, standby INTEGER, updated_at TEXT, sensor_json TEXT)')
        c.execute('CREATE TABLE IF NOT EXISTS actions (id TEXT PRIMARY KEY, mine_id TEXT, created_at TEXT, state TEXT, reason TEXT, actor TEXT, payload TEXT, actual_recovered_mt REAL, actual_cost REAL)')
        c.execute('CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, action_id TEXT, at TEXT, state TEXT, actor TEXT, reason TEXT)')
        c.execute('CREATE TABLE IF NOT EXISTS field_results (id INTEGER PRIMARY KEY, at TEXT, site_id TEXT, assay_pct REAL, notes TEXT)')
        c.execute('CREATE TABLE IF NOT EXISTS site_inputs (mine_id TEXT PRIMARY KEY, payload TEXT, updated_at TEXT)')
        c.execute('CREATE TABLE IF NOT EXISTS machine_inputs (machine_id TEXT PRIMARY KEY, payload TEXT, updated_at TEXT)')
        c.execute('CREATE TABLE IF NOT EXISTS shift_logs (mine_id TEXT, day TEXT, shift TEXT, payload TEXT, updated_at TEXT, PRIMARY KEY(mine_id,day,shift))')
        c.execute('CREATE TABLE IF NOT EXISTS sensor_readings (id INTEGER PRIMARY KEY, mine_id TEXT, payload TEXT, created_at TEXT)')
        c.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)')
        if not c.execute("SELECT 1 FROM metadata WHERE key='standby_seed_v2'").fetchone():
            for r in c.execute('SELECT id FROM machines').fetchall():
                standby=int(hashlib.sha1(r['id'].encode()).hexdigest()[:8],16)%5==0
                c.execute('UPDATE machines SET standby=? WHERE id=?',(standby,r['id']))
            c.execute("INSERT INTO metadata VALUES ('standby_seed_v2','hash-based; simulated')")
        if c.execute('SELECT count(*) FROM machines').fetchone()[0]:return
        mines=[m for m in catalog() if m['target']==1 and m['country']=='India']
        if not mines:raise ValueError('No India occurrences available for labelled demo assignments')
        d=pd.read_csv(DATA/'ai4i2020_test_FINAL.csv')
        for i,row in d.iterrows():
            mine=mines[i%len(mines)]; uid=f"EQ-{int(row['UDI']):05d}"
            offset=(int(hashlib.sha1(uid.encode()).hexdigest()[:5],16)%1000-500)/100000
            standby=int(hashlib.sha1(uid.encode()).hexdigest()[:8],16)%5==0
            c.execute('INSERT INTO machines VALUES (?,?,?,?,?,?,?)',(uid,mine['id'],mine['latitude']+offset,mine['longitude']-offset,standby,now(),json.dumps(row.to_dict())))

def reserved_machines():
    with connection() as c:
        return {json.loads(r['payload']).get('machine_id') for r in c.execute("SELECT payload FROM actions WHERE state='accepted'")}

def machine_rows():
    with connection() as c:return [dict(r) for r in c.execute('SELECT * FROM machines ORDER BY id')]

def save_action(mine_id,payload):
    # Stable per recommendation/input snapshot: repeated requests keep operator decisions.
    uid='ACT-'+hashlib.sha256(json.dumps([mine_id,payload],sort_keys=True).encode()).hexdigest()[:16]
    with connection() as c:
        c.execute('INSERT OR IGNORE INTO actions VALUES (?,?,?,?,?,?,?,?,?)',(uid,mine_id,now(),'pending','','',json.dumps(payload),None,None))
        r=dict(c.execute('SELECT * FROM actions WHERE id=?',(uid,)).fetchone())
    return dict(id=uid,state=r['state'],**payload)

def update_action(uid,state,reason,actor,recovered=None,cost=None):
    with connection() as c:
        c.execute('BEGIN IMMEDIATE')
        row=c.execute('SELECT * FROM actions WHERE id=?',(uid,)).fetchone()
        if not row:raise KeyError(uid)
        allowed={'pending':{'acknowledged','accepted','rejected'},'acknowledged':{'accepted','rejected'},'accepted':{'completed','rejected'},'rejected':set(),'completed':set()}
        if state not in allowed[row['state']]:raise ValueError(f"Cannot change {row['state']} to {state}")
        payload=json.loads(row['payload'])
        if state=='accepted':
            others=c.execute("SELECT payload FROM actions WHERE mine_id=? AND state='accepted' AND id!=?",(row['mine_id'],uid)).fetchall()
            new_slots={(r['date'],r['shift']) for r in payload.get('recovery_schedule',[])}
            if any(json.loads(o['payload']).get('start_date')==payload.get('start_date') or new_slots.intersection({(r['date'],r['shift']) for r in json.loads(o['payload']).get('recovery_schedule',[])}) for o in others):
                raise ValueError('An alternative is already accepted for this mine and start date. Resolve it before approving another; recoveries cannot be added.')
        if state=='accepted' and payload.get('machine_id'):
            machine=payload['machine_id']
            others=c.execute("SELECT id,payload FROM actions WHERE state='accepted' AND id!=?",(uid,)).fetchall()
            if any(json.loads(o['payload']).get('machine_id')==machine for o in others):raise ValueError('Machine is reserved by another accepted action')
        if state=='completed' and (recovered is None or cost is None):raise ValueError('Completion needs actual recovery and actual cost')
        c.execute('UPDATE actions SET state=?,reason=?,actor=?,actual_recovered_mt=?,actual_cost=? WHERE id=?',(state,reason,actor,recovered,cost,uid))
        c.execute('INSERT INTO events(action_id,at,state,actor,reason) VALUES (?,?,?,?,?)',(uid,now(),state,actor,reason))
    return dict(id=uid,state=state,reason=reason)

def records():
    with connection() as c:
        rows=[dict(r) for r in c.execute('SELECT * FROM actions ORDER BY created_at DESC LIMIT 500')]
    for r in rows:r['payload']=json.loads(r['payload'])
    return rows


def site_inputs(uid):
    with connection() as c:r=c.execute('SELECT payload FROM site_inputs WHERE mine_id=?',(uid,)).fetchone()
    return json.loads(r['payload']) if r else {}

def machine_inputs():
    with connection() as c:return {r['machine_id']:json.loads(r['payload']) for r in c.execute('SELECT * FROM machine_inputs')}

def shift_logs(uid):
    with connection() as c:return [json.loads(r['payload']) for r in c.execute('SELECT payload FROM shift_logs WHERE mine_id=? ORDER BY day,shift',(uid,))]
