#!/usr/bin/env python3
"""
NICK Blueprint Miner v2 — SAVI SQLite + rxn: format
=====================================================
Output MUST use rxn:* reaction-formatted molecules from the SAVI
combinatorial SQLite DB. Raw SMILES are REJECTED by the validator.
"""
import json, os, sys, time, random, math, sqlite3
from pathlib import Path

INPUT_FILE  = '/workspace/input.json'
OUTPUT_FILE = '/output/result.json'
os.makedirs('/tmp/nick_bp', exist_ok=True)
os.makedirs('/output', exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_input():
    try:
        with open(INPUT_FILE) as f: return json.load(f)
    except Exception as e:
        log(f"input.json error: {e}")
        return {}

def extract_params(data):
    target = (data.get('target_sequence') or data.get('target') or
              data.get('target_protein') or data.get('weekly_target') or 'P23977')
    if isinstance(target, dict):
        target = target.get('sequence') or target.get('id') or 'P23977'
    ats = data.get('antitargets', data.get('antitarget', []))
    if isinstance(ats, str): ats = [ats]
    aw = float(data.get('antitarget_weight', 0.9))
    tl = int(data.get('time_limit', data.get('time_budget', 1800)))
    nm = int(data.get('num_molecules', 100))
    db = (data.get('db_path') or data.get('database') or data.get('savi_db'))
    if not db:
        for pat in ['*.db', '*.sqlite', '*.sqlite3']:
            hits = list(Path('/workspace').glob(pat))
            if hits: db = str(hits[0]); break
    log(f"Target:{str(target)[:40]} ATs:{len(ats)} AW:{aw} T:{tl}s N:{nm} DB:{db}")
    return target, ats, aw, tl, nm, db

KNOWN_SEQS = {
    'P23977': (
        'MASKKMNESRNFIQPRTFGSMPKTLSSSKSPRDEQVHKKKSKEAKGPSGTHIQHNTRSITEEQPTNSVMHILQ'
        'NLSRLNEPQKTQVPQHLPKHNTKNHKLLILKIFIPMMILSLSVNLFPLFISFSYFFLLIKFITSPFLHQTLYF'
        'VLFGLSSFLVAVLSAVVLLAQDYQVSNISNLIPQLNELYESAAIPPPKPDLTPKNQATPRGCLESFLKLFFNL'
        'FGMIPYMILICFLQLGLFVHFGLMSAQTLNSSPAFKQAIQETYQFSRTLQYVLSELLKSIIVVVLATIIFGFL'
        'NLAAYLMGQLQNMDMHPSEGPRNLKRLNPPAVSHEPIQSQKMKESTDDTEGGISRISGSGKLMSRPNEAGNAE'
        'DDEATRLILKLREIQEYLIEQHGALISLGFVVNILQPIMVFAGMTSHGRYQDIMGLPFPKAFEIPYQSLRLGK'
        'VDAKMISTVQGILLKQLMAVSACLAGLASLFAIAMDTLVNASEFNLDTLNFYIIMQVLSANVTFMIVSKFWDN'
        'FTLHSLYYLIGCYLSGYVTATPTLNIPIFVYLAFKAGPKLRMMFTEESYALGSSCSNLQCYIGQQPKKLHDQL'
        'SCGKEGFAEQLIAMTQFMQDYTPETTNHMSSSDLTAAQLHVFNAKAIEMKHQ'),
}

def resolve_seq(v):
    s = str(v).strip()
    if len(s) <= 15 and s.replace('_','').isalnum():
        q = KNOWN_SEQS.get(s.upper())
        if q: return q
    return s

class SAVIDatabase:
    def __init__(self, db_path):
        self.conn = None; self.mol_table = None
        self.rxn_col = None; self.smiles_col = None; self.n_mols = 0
        if db_path and os.path.exists(db_path):
            try:
                self.conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1",
                                            uri=True, check_same_thread=False)
                tables = [r[0] for r in self.conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
                log(f"DB tables: {tables}")
                for t in tables:
                    cols = [r[1] for r in self.conn.execute(f"PRAGMA table_info({t})").fetchall()]
                    log(f"  {t}: {cols}")
                    cl = [c.lower() for c in cols]
                    if any(x in ' '.join(cl) for x in ['rxn','mol','smiles','molecule']):
                        self.mol_table = t
                        for c in cols:
                            if any(x in c.lower() for x in ['rxn','mol_id','id']):
                                self.rxn_col = c
                            if 'smiles' in c.lower():
                                self.smiles_col = c
                if self.mol_table:
                    self.n_mols = self.conn.execute(f"SELECT COUNT(*) FROM {self.mol_table}").fetchone()[0]
                    log(f"Mol table:{self.mol_table}({self.n_mols:,}) rxn_col:{self.rxn_col} smi_col:{self.smiles_col}")
            except Exception as e:
                log(f"DB error: {e}")

    def ok(self): return self.conn is not None and self.mol_table is not None

    def sample(self, n=500):
        if not self.ok(): return []
        off = random.randint(0, max(0, self.n_mols - n))
        cols = self.rxn_col
        if self.smiles_col: cols += ', ' + self.smiles_col
        try:
            rows = self.conn.execute(f"SELECT {cols} FROM {self.mol_table} LIMIT {n} OFFSET {off}").fetchall()
            result = []
            for r in rows:
                rxn = str(r[0])
                if not rxn.startswith('rxn:'): rxn = 'rxn:' + rxn
                smi = str(r[1]) if self.smiles_col and len(r) > 1 else None
                result.append((rxn, smi))
            return result
        except Exception as e:
            log(f"sample err:{e}"); return []

    def smiles_for(self, rxn_id):
        if not self.ok() or not self.smiles_col: return None
        lid = rxn_id[4:] if rxn_id.startswith('rxn:') else rxn_id
        for qid in [lid, rxn_id]:
            try:
                r = self.conn.execute(f"SELECT {self.smiles_col} FROM {self.mol_table} WHERE {self.rxn_col}=?", (qid,)).fetchone()
                if r: return str(r[0])
            except: pass
        return None

    def neighbours(self, rxn_id, n=5):
        parts = rxn_id.replace('rxn:','').split(':')
        result = []
        if len(parts) >= 2:
            tmpl, r1 = parts[0], parts[1]
            try:
                base = int(r1)
                for off in random.sample(range(-100,101), min(n*3,201)):
                    nr = base + off
                    if nr >= 0:
                        tail = ':'.join([tmpl, str(nr)] + parts[2:])
                        result.append('rxn:' + tail)
                        if len(result) >= n: break
            except: pass
            if self.ok():
                try:
                    rows = self.conn.execute(
                        f"SELECT {self.rxn_col} FROM {self.mol_table} WHERE {self.rxn_col} LIKE ? LIMIT ?",
                        (f"{tmpl}:%", n)).fetchall()
                    for row in rows:
                        rx = str(row[0])
                        if not rx.startswith('rxn:'): rx = 'rxn:'+rx
                        if rx != rxn_id: result.append(rx)
                except: pass
        return result[:n]

    def close(self):
        if self.conn: self.conn.close()

class Scorer:
    def __init__(self):
        self.pw = None
        for p in ['/root/nova','/nova','/app','/workspace','.','/root']:
            if os.path.isdir(os.path.join(p,'PSICHIC')):
                if p not in sys.path: sys.path.insert(0,p)
                break
        try:
            os.environ.setdefault('DEVICE_OVERRIDE','cpu')
            from PSICHIC.wrapper import PsichicWrapper
            self.pw = PsichicWrapper()
            log("PSICHIC loaded")
        except Exception as e: log(f"PSICHIC: {e}")

    def score(self, target_seq, antitargets, smiles_list, aw=0.9):
        if not self.pw or not smiles_list: return {}
        try:
            self.pw.run_challenge_start(target_seq)
            df = self.pw.run_validation(smiles_list)
            ts = {str(r['Ligand']): float(r.get('predicted_binding_affinity',0)) for _,r in df.iterrows()}
        except Exception as e: log(f"target score err:{e}"); ts = {}
        at = {s:0.0 for s in smiles_list}
        for seq in antitargets:
            if not seq or len(str(seq))<20: continue
            try:
                self.pw.run_challenge_start(seq)
                df = self.pw.run_validation(smiles_list)
                for _,r in df.iterrows():
                    s=str(r['Ligand']); v=float(r.get('predicted_binding_affinity',0))
                    at[s]=max(at.get(s,0),v)
            except Exception as e: log(f"at score:{e}")
        return {s: ts.get(s,0)-aw*at.get(s,0) for s in smiles_list}

class ComponentRanker:
    def __init__(self, alpha=0.3):
        self.alpha=alpha; self.tmpl={}; self.react={}
    def update(self, rxn_id, sc):
        parts = rxn_id.replace('rxn:','').split(':')
        for i,p in enumerate(parts):
            d = self.tmpl if i==0 else self.react
            d[p] = self.alpha*sc+(1-self.alpha)*d[p] if p in d else sc
    def score(self, rxn_id):
        parts = rxn_id.replace('rxn:','').split(':')
        scores = [self.tmpl.get(parts[0],0.0)] + [self.react.get(p,0.0) for p in parts[1:]]
        return sum(scores)/len(scores) if scores else 0.0
    def top_tmpls(self, n=5): return sorted(self.tmpl.items(), key=lambda x:-x[1])[:n]

def div_select(scored_triples, n=100):
    # scored_triples: [(rxn_id, score, smiles_or_None)]
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors, DataStructs
        valid, invalid = [], []
        for rxn,sc,smi in scored_triples:
            m = Chem.MolFromSmiles(smi) if smi else None
            if m: valid.append((rxn,sc,rdMolDescriptors.GetMACCSKeysFingerprint(m)))
            else: invalid.append(rxn)
        valid.sort(key=lambda x:-x[1])
        sel, fps = [], []
        for rxn,sc,fp in valid:
            if not sel:
                sel.append(rxn); fps.append(fp); continue
            sims = DataStructs.BulkTanimotoSimilarity(fp, fps)
            if not sims or max(sims)<0.80:
                sel.append(rxn); fps.append(fp)
            if len(sel)>=n: break
        if len(sel)<n:
            used=set(sel)
            for rxn,_,_ in valid:
                if rxn not in used: sel.append(rxn)
                if len(sel)>=n: break
            for rxn in invalid:
                if rxn not in used: sel.append(rxn)
                if len(sel)>=n: break
        return sel[:n]
    except:
        return [r for r,_,_ in sorted(scored_triples,key=lambda x:-x[1])[:n]]

def write_out(rxn_ids, note=""):
    valid=[r for r in rxn_ids if r and r.startswith('rxn:')]
    tmp=OUTPUT_FILE+'.tmp'
    try:
        with open(tmp,'w') as f: json.dump({"molecules":valid,"n":len(valid),"note":note},f)
        os.replace(tmp,OUTPUT_FILE)
        log(f"Output: {len(valid)} rxn: mols ({note})")
    except Exception as e: log(f"write err:{e}")

def main():
    t0=time.time()
    log("="*60); log("NICK Blueprint Miner v2"); log("="*60)
    data=load_input()
    tgt_raw,ats_raw,aw,tlimit,nmols,db_path=extract_params(data)
    tgt_seq=resolve_seq(tgt_raw)
    at_seqs=[resolve_seq(a) for a in ats_raw if a]
    at_seqs=[s for s in at_seqs if len(str(s))>10]

    db=SAVIDatabase(db_path) if db_path else SAVIDatabase(None)
    if not db.ok():
        log("FATAL: No SAVI DB found"); write_out([],"no_db"); return

    # Sample + write immediately
    log("Initial sample...")
    init=db.sample(n=500)
    init_rxn=[r for r,_ in init]
    write_out(init_rxn[:nmols],"initial")

    scorer=Scorer()
    if not scorer.pw or len(str(tgt_seq))<20:
        log("No PSICHIC/target — template diversity fallback")
        tmpls={}
        for rxn,_ in init:
            t=rxn.replace('rxn:','').split(':')[0]
            tmpls.setdefault(t,[]).append(rxn)
        diverse=[]
        for t in list(tmpls.keys()):
            diverse.append(tmpls[t][0])
            if len(diverse)>=nmols: break
        write_out(diverse,"template_diverse"); return

    # Get SMILES + score initial
    smi_map={}
    for rxn,smi in init:
        if smi: smi_map[rxn]=smi
        else:
            s=db.smiles_for(rxn)
            if s: smi_map[rxn]=s

    all_scored={}
    ranker=ComponentRanker()

    if smi_map:
        log(f"Scoring {len(smi_map)} initial mols...")
        sel=scorer.score(tgt_seq,at_seqs,list(smi_map.values()),aw)
        for rxn,smi in smi_map.items():
            sc=sel.get(smi,0.0); all_scored[rxn]=(sc,smi); ranker.update(rxn,sc)
        for rxn,_ in init:
            if rxn not in all_scored: all_scored[rxn]=(0.0,None)
        if sel:
            top5=sorted(sel.items(),key=lambda x:-x[1])[:5]
            log(f"Init top5:{[round(v,3) for _,v in top5]} mean:{sum(sel.values())/len(sel):.4f}")
            trips=[(r,v[0],v[1]) for r,v in all_scored.items()]
            write_out(div_select(trips,nmols),"init_scored")

    # DJA optimization
    remaining=tlimit-(time.time()-t0)-30
    if remaining<60:
        log("Not enough time for optimization")
    else:
        log(f"DJA optimization ({remaining:.0f}s)...")
        best=max((v for v,_ in all_scored.values()),default=0.0)
        pop_A=sorted(all_scored,key=lambda x:-all_scored[x][0])[:200]
        pop_B=sorted(all_scored,key=lambda x:-all_scored[x][0])[200:250]
        tabu=set(all_scored.keys())
        no_improve=0; iteration=0

        while time.time()-t0 < t0+tlimit-30:
            if time.time()-t0 > tlimit-30: break
            try:
                cands=[]
                scores_A={k:all_scored[k][0] for k in pop_A if k in all_scored}
                if scores_A:
                    best_A=max(scores_A,key=lambda x:scores_A[x])
                    cands+=db.neighbours(best_A,n=10)
                    for tmpl,_ in ranker.top_tmpls(3):
                        cands+=db.neighbours(f"rxn:{tmpl}:0",n=5)
                for rxn in random.sample(pop_B,min(5,len(pop_B))):
                    cands+=db.neighbours(rxn,n=3)
                if no_improve>=5:
                    log(f"Anti-plateau iter {iteration}")
                    cands+=[r for r,_ in db.sample(n=50)]
                cands=[c for c in set(cands) if c not in tabu and c not in all_scored][:50]
                tabu.update(cands)
                if not cands:
                    cands=[r for r,_ in db.sample(n=20) if r not in all_scored]
                    tabu.update(cands)
                if not cands: time.sleep(2); iteration+=1; continue

                # Score batch
                smi_batch={}
                for rxn in cands[:25]:
                    s=db.smiles_for(rxn)
                    if s: smi_batch[rxn]=s
                if smi_batch:
                    sel=scorer.score(tgt_seq,at_seqs,list(smi_batch.values()),aw)
                    for rxn,smi in smi_batch.items():
                        sc=sel.get(smi,ranker.score(rxn))
                        all_scored[rxn]=(sc,smi); ranker.update(rxn,sc)
                for rxn in cands:
                    if rxn not in all_scored: all_scored[rxn]=(ranker.score(rxn),None)

                cur_best=max((v for v,_ in all_scored.values()),default=0.0)
                if cur_best>best:
                    best=cur_best; no_improve=0
                    log(f"New best:{best:.4f}")
                else:
                    no_improve+=1

                # Update pops
                new_sorted=sorted(all_scored.items(),key=lambda x:-x[1][0])[:15]
                for rxn,(sc,_) in new_sorted:
                    if rxn not in pop_A:
                        if len(pop_A)<200: pop_A.append(rxn)
                        else:
                            worst=min(pop_A,key=lambda x:all_scored.get(x,(0,))[0])
                            if sc>all_scored.get(worst,(0,))[0]:
                                pop_A.remove(worst); pop_A.append(rxn)
                if iteration%10==0 and pop_A and pop_B:
                    n=min(5,len(pop_A),len(pop_B))
                    wA=sorted(pop_A,key=lambda x:all_scored.get(x,(0,))[0])[:n]
                    bB=sorted(pop_B,key=lambda x:-all_scored.get(x,(0,))[0])[:n]
                    for wa,bb in zip(wA,bB):
                        if wa in pop_A: pop_A.remove(wa)
                        if bb in pop_B: pop_B.remove(bb)
                        pop_A.append(bb); pop_B.append(wa)

                trips=[(r,v[0],v[1]) for r,v in all_scored.items()]
                write_out(div_select(trips,nmols),"optimizing")
                log(f"Iter {iteration}: {len(all_scored)} scored, best={best:.4f}, t={time.time()-t0:.0f}s")
            except Exception as e:
                log(f"opt err:{e}")
                import traceback; traceback.print_exc()
            iteration+=1

        # Final write
        trips=[(r,v[0],v[1]) for r,v in all_scored.items()]
        final=div_select(trips,nmols)
        mean=sum(all_scored[m][0] for m in final if m in all_scored)/max(1,len(final))
        write_out(final,f"final_mean_{mean:.4f}")
        log(f"Done: {len(final)} rxn: mols, mean={mean:.4f}")

    db.close()
    log(f"Total time: {time.time()-t0:.0f}s")

if __name__=='__main__':
    main()
