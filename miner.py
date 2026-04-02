"""
NICK Blueprint Miner v3 - SN68 Nova Optimized
============================================================
Key fixes vs v2:
1. RESTRICT to rxn:1 + rxn:2 ONLY (validator hardcoded constraint)
2. BEST molecule FIRST in output (num_molecules_boltz:1, sample_selection:first)
3. Heavy-atom-aware scoring (smaller + high-affinity = better Boltz score)
4. DAT seed similarity heuristic when PSICHIC unavailable
5. All neighbour exploration stays within rxn:1/rxn:2 only
"""
import json, os, sys, time, random, math, sqlite3
from pathlib import Path

INPUT_FILE  = '/workspace/input.json'
OUTPUT_FILE = '/output/result.json'
os.makedirs('/tmp/nick_bp', exist_ok=True)
os.makedirs('/output', exist_ok=True)

# Known DAT binders from validator reinvent_seeds (methylphenidate analogues for P23977)
DAT_SEEDS = [
    'COC(=O)C1(c2cccc2)CC[NH+](C)CC1',
    'Fc1ccc(C2C[NH+](CCc3ccccc3)CC2)cc1',
    'COc1ccc(C2CCNCC2)cc1OC',
    'O=C(NCCN1CCCC1)c1ccc(Cl)cc1',
    'c1ccc2c(c1)C(CN1CCCC1)CC2',
]

# Validator validity.py HARDCODES only these two templates as allowed
ALLOWED_TEMPLATES = {1, 2}

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
    aw  = float(data.get('antitarget_weight', 0.9))
    tl  = int(data.get('time_limit', data.get('time_budget', 1800)))
    nm  = int(data.get('num_molecules', 30))
    db  = (data.get('db_path') or data.get('database') or data.get('savi_db'))
    if not db:
        for pat in ['*.db', '*.sqlite', '*.sqlite3']:
            hits = list(Path('/workspace').glob(pat))
            if hits: db = str(hits[0]); break
    log(f"Target:{str(target)[:40]} ATs:{len(ats)} AW:{aw} T:{tl}s N:{nm} DB:{db}")
    return target, ats, aw, tl, nm, db

KNOWN_SEQS = {
    'P23977': (
        'MASKKMNESNFIQPRTFGSMPKTLSSSKSPRDEQVHKKKSKEAKGPSGTHIQHNTRSITEEQPTNSVMHILQ'
        'NLSRLNEPQKTQVPQHLPKHNTKNHKLLILKIFIPMMILSLSVNLFPLFISFSYFFLLIKFITSPFLHQTLYF'
        'VLFGLSSFLVAVLSAVVLLAQDYQVSNISNLIQLQNELYESAAIPPKPDLTPKNQATPRGCLESFLKLFFNL'
        'FGMIPYMILICFLQLGLFVHFGLMSAQTLNSSPAFKQAIQETYQFSRTLQYVLSELLKSIIVVVLATIIFGFL'
        'NLAAYLMGQLQNMDMHPSEGPRNLKRLNPPAVSHEPIQSQKMKESTDDTEGGISRISGSGKLMSRPNEAGNAE'
        'DDEATRLILKLREIQEYIIEQHGALISLGFVVNILQPIMVFAGMTSHGRYQDIMGLPFPKAFEIPYQSLRLGK'
        'VDAKMISTVQGILLKQLMAVSACLAGLASLFAIANDTLVNASEFNLDTLNFYIIMQVLSANVTFMIVSKFWDN'
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
                    log(f"Mol table:{self.mol_table}({self.n_mols:,}) rxn:{self.rxn_col} smi:{self.smiles_col}")
            except Exception as e:
                log(f"DB error: {e}")

    def ok(self): return self.conn is not None and self.mol_table is not None

    def sample_rxn12(self, n=500):
        """Sample ONLY from rxn:1 and rxn:2 — the only templates validator allows."""
        if not self.ok(): return []
        result = []
        cols = self.rxn_col
        if self.smiles_col: cols += ', ' + self.smiles_col
        try:
            half = max(n // 2, 1)
            for tmpl_num in [1, 2]:
                like_pat = f"rxn:{tmpl_num}:%"
                rows = self.conn.execute(
                    f"SELECT {cols} FROM {self.mol_table} "
                    f"WHERE {self.rxn_col} LIKE ? ORDER BY RANDOM() LIMIT ?",
                    (like_pat, half)).fetchall()
                for r in rows:
                    rxn = str(r[0])
                    if not rxn.startswith('rxn:'): rxn = 'rxn:' + rxn
                    smi = str(r[1]) if self.smiles_col and len(r) > 1 else None
                    result.append((rxn, smi))
            random.shuffle(result)
            return result
        except Exception as e:
            log(f"sample_rxn12 err:{e}"); return self.sample(n)

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
                r = self.conn.execute(
                    f"SELECT {self.smiles_col} FROM {self.mol_table} WHERE {self.rxn_col}=?",
                    (qid,)).fetchone()
                if r: return str(r[0])
            except: pass
        return None

    def neighbours_rxn12(self, rxn_id, n=5):
        """Get neighbours but STAY within rxn:1 and rxn:2 only."""
        parts = rxn_id.replace('rxn:','').split(':')
        result = []
        if len(parts) >= 2:
            tmpl, r1 = parts[0], parts[1]
            try:
                if int(tmpl) not in ALLOWED_TEMPLATES:
                    # Redirect to allowed template
                    tmpl = str(random.choice([1, 2]))
            except: tmpl = str(random.choice([1, 2]))
            try:
                base = int(r1)
                offsets = random.sample(range(-100, 101), min(n*3, 201))
                for off in offsets:
                    nr = base + off
                    if nr >= 0:
                        tail = ':'.join([tmpl, str(nr)] + parts[2:])
                        result.append('rxn:' + tail)
                    if len(result) >= n: break
            except: pass
        # Also do DB lookup for same template
        if self.ok() and parts:
            try:
                tmpl_prefix = f"rxn:{parts[0]}:%"
                rows = self.conn.execute(
                    f"SELECT {self.rxn_col} FROM {self.mol_table} "
                    f"WHERE {self.rxn_col} LIKE ? ORDER BY RANDOM() LIMIT ?",
                    (tmpl_prefix, n)).fetchall()
                for row in rows:
                    rx = str(row[0])
                    if not rx.startswith('rxn:'): rx = 'rxn:'+rx
                    if rx != rxn_id: result.append(rx)
            except: pass
        return result[:n]

    def close(self):
        if self.conn: self.conn.close()


class Scorer:
    """PSICHIC-based scorer with graceful fallback."""
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
        except Exception as e: log(f"PSICHIC unavailable: {e}")

    def score(self, target_seq, antitargets, smiles_list, aw=0.9):
        if not self.pw or not smiles_list: return {}
        try:
            self.pw.run_challenge_start(target_seq)
            df = self.pw.run_validation(smiles_list)
            ts = {str(r['Ligand']): float(r.get('predicted_binding_affinity',0))
                  for _,r in df.iterrows()}
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


def seed_similarity_score(smi, seed_fps=None):
    """
    Heuristic when PSICHIC unavailable:
    Tanimoto similarity to known DAT binders + small-HA bonus.
    Rationale: heavy_atom_normalization rewards smaller molecules;
    DAT_SEEDS represent known good chemical space for P23977.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors, DataStructs
        m = Chem.MolFromSmiles(smi)
        if not m: return 0.0
        ha = m.GetNumHeavyAtoms()
        if ha < 10 or ha > 40: return 0.0

        fp = rdMolDescriptors.GetMACCSKeysFingerprint(m)
        if seed_fps is None:
            seed_fps = []
            for s in DAT_SEEDS:
                sm = Chem.MolFromSmiles(s)
                if sm: seed_fps.append(rdMolDescriptors.GetMACCSKeysFingerprint(sm))
        if not seed_fps: return 0.5

        sims = DataStructs.BulkTanimotoSimilarity(fp, seed_fps)
        max_sim = max(sims) if sims else 0.0
        # Heavier molecules penalised (mimics heavy_atom_normalization)
        ha_factor = 20.0 / max(ha, 10)
        return max_sim * ha_factor
    except Exception as e:
        log(f"seed_sim err:{e}"); return 0.0


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
    """Diversity-filtered selection by MACCS Tanimoto. Returns rxn_ids."""
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors, DataStructs
        valid, invalid = [], []
        for rxn,sc,smi in scored_triples:
            m = Chem.MolFromSmiles(smi) if smi else None
            if m: valid.append((rxn,sc,rdMolDescriptors.GetMACCSKeysFingerprint(m)))
            else: invalid.append((rxn,sc))
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
            for rxn,_ in invalid:
                if rxn not in used: sel.append(rxn)
                if len(sel)>=n: break
        return sel[:n]
    except:
        return [r for r,_,_ in sorted(scored_triples,key=lambda x:-x[1])[:n]]


def best_first_output(all_scored, nmols):
    """
    Critical: put the HIGHEST-SCORING molecule FIRST.
    Validator uses sample_selection:first + num_molecules_boltz:1,
    meaning only molecule #1 gets Boltz-evaluated.
    Fill remaining slots with diversity-selected molecules.
    """
    trips = [(r,v[0],v[1]) for r,v in all_scored.items() if r and r.startswith('rxn:')]
    if not trips: return []
    trips_sorted = sorted(trips, key=lambda x:-x[1])
    best_rxn = trips_sorted[0][0]
    if len(trips_sorted) > 1:
        rest = div_select(trips_sorted[1:], nmols-1)
    else:
        rest = []
    return [best_rxn] + rest


def write_out(rxn_ids, note=""):
    valid = [r for r in rxn_ids if r and r.startswith('rxn:')]
    # Deduplicate while preserving order
    seen = set(); deduped = []
    for r in valid:
        if r not in seen: seen.add(r); deduped.append(r)
    tmp = OUTPUT_FILE+'.tmp'
    try:
        with open(tmp,'w') as f:
            json.dump({"molecules":deduped,"n":len(deduped),"note":note},f)
        os.replace(tmp,OUTPUT_FILE)
        log(f"Output: {len(deduped)} rxn: mols [{note}]")
    except Exception as e: log(f"write err:{e}")


def main():
    t0=time.time()
    log("="*60)
    log("NICK Blueprint Miner v3 — SN68 Nova Optimized")
    log("="*60)

    data=load_input()
    tgt_raw,ats_raw,aw,tlimit,nmols,db_path=extract_params(data)
    tgt_seq=resolve_seq(tgt_raw)
    at_seqs=[resolve_seq(a) for a in ats_raw if a]
    at_seqs=[s for s in at_seqs if len(str(s))>10]

    db=SAVIDatabase(db_path) if db_path else SAVIDatabase(None)
    if not db.ok():
        log("FATAL: No SAVI DB found"); write_out([],"no_db"); return

    # ── PHASE 1: Initial sample rxn:1 + rxn:2 ONLY ──────────────────────────
    log("Phase 1: Sampling rxn:1+rxn:2 only...")
    init=db.sample_rxn12(n=600)
    init_rxn=[r for r,_ in init]
    # Immediate fallback write in case we crash later
    write_out(init_rxn[:nmols], "initial_rxn12_fallback")

    # ── PHASE 2: Score initial sample ────────────────────────────────────────
    scorer=Scorer()

    # Build smi_map
    smi_map={}
    for rxn,smi in init:
        if smi: smi_map[rxn]=smi
        else:
            s=db.smiles_for(rxn)
            if s: smi_map[rxn]=s

    all_scored={}
    ranker=ComponentRanker()

    # Pre-compute seed fingerprints for heuristic scoring
    seed_fps=None
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
        seed_fps=[]
        for s in DAT_SEEDS:
            m=Chem.MolFromSmiles(s)
            if m: seed_fps.append(rdMolDescriptors.GetMACCSKeysFingerprint(m))
        log(f"Seed fingerprints: {len(seed_fps)}")
    except Exception as e: log(f"RDKit not available: {e}")

    if smi_map:
        if scorer.pw:
            log(f"PSICHIC scoring {len(smi_map)} initial mols...")
            sel=scorer.score(tgt_seq,at_seqs,list(smi_map.values()),aw)
            for rxn,smi in smi_map.items():
                sc=sel.get(smi,0.0)
                all_scored[rxn]=(sc,smi); ranker.update(rxn,sc)
            if sel:
                top5=sorted(sel.items(),key=lambda x:-x[1])[:5]
                log(f"Init top5 scores:{[round(v,3) for _,v in top5]}")
        else:
            log("Using DAT seed-similarity heuristic (no PSICHIC)...")
            for rxn,smi in smi_map.items():
                sc=seed_similarity_score(smi,seed_fps)
                all_scored[rxn]=(sc,smi); ranker.update(rxn,sc)
            scored_vals=[v for v,_ in all_scored.values() if v>0]
            if scored_vals:
                log(f"Heuristic top score:{max(scored_vals):.4f} mean:{sum(scored_vals)/len(scored_vals):.4f}")

    for rxn,_ in init:
        if rxn not in all_scored: all_scored[rxn]=(0.0,None)

    # Write best-first after initial scoring
    write_out(best_first_output(all_scored,nmols), "init_scored_best_first")

    # ── PHASE 3: DJA Optimisation (rxn:1+rxn:2 only) ────────────────────────
    remaining=tlimit-(time.time()-t0)-30
    if remaining<60:
        log("Insufficient time for DJA optimisation")
    else:
        log(f"DJA optimisation ({remaining:.0f}s available)...")
        best=max((v for v,_ in all_scored.values()),default=0.0)
        pop_A=sorted(all_scored,key=lambda x:-all_scored[x][0])[:200]
        pop_B=sorted(all_scored,key=lambda x:-all_scored[x][0])[200:250]
        tabu=set(all_scored.keys())
        no_improve=0; iteration=0

        while time.time()-t0 < tlimit-30:
            try:
                cands=[]
                scores_A={k:all_scored[k][0] for k in pop_A if k in all_scored}
                if scores_A:
                    best_A=max(scores_A,key=lambda x:scores_A[x])
                    cands+=db.neighbours_rxn12(best_A,n=10)
                    for tmpl,_ in ranker.top_tmpls(3):
                        # Only explore allowed templates
                        if tmpl in ['1','2']:
                            cands+=db.neighbours_rxn12(f"rxn:{tmpl}:0",n=5)
                for rxn in random.sample(pop_B,min(5,len(pop_B))):
                    cands+=db.neighbours_rxn12(rxn,n=3)
                if no_improve>=5:
                    log(f"Anti-plateau at iter {iteration}")
                    cands+=[r for r,_ in db.sample_rxn12(n=50)]

                cands=[c for c in set(cands) if c not in tabu and c not in all_scored][:50]
                tabu.update(cands)
                if not cands:
                    cands=[r for r,_ in db.sample_rxn12(n=20) if r not in all_scored]
                    tabu.update(cands)
                if not cands: time.sleep(2); iteration+=1; continue

                # Score batch of up to 25
                smi_batch={}
                for rxn in cands[:25]:
                    s=db.smiles_for(rxn)
                    if s: smi_batch[rxn]=s

                if smi_batch:
                    if scorer.pw:
                        sel=scorer.score(tgt_seq,at_seqs,list(smi_batch.values()),aw)
                        for rxn,smi in smi_batch.items():
                            sc=sel.get(smi,ranker.score(rxn))
                            all_scored[rxn]=(sc,smi); ranker.update(rxn,sc)
                    else:
                        for rxn,smi in smi_batch.items():
                            sc=seed_similarity_score(smi,seed_fps)
                            all_scored[rxn]=(sc,smi); ranker.update(rxn,sc)

                for rxn in cands:
                    if rxn not in all_scored: all_scored[rxn]=(ranker.score(rxn),None)

                cur_best=max((v for v,_ in all_scored.values()),default=0.0)
                if cur_best>best:
                    best=cur_best; no_improve=0
                    log(f"New best:{best:.4f} at iter {iteration}")
                else:
                    no_improve+=1

                # Update population
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

                # Periodic best-first write
                if iteration%5==0:
                    write_out(best_first_output(all_scored,nmols),
                              f"opt_iter={iteration} best={best:.4f}")
                    log(f"Iter {iteration}: {len(all_scored)} scored best={best:.4f} t={time.time()-t0:.0f}s")

            except Exception as e:
                log(f"opt err:{e}")
                import traceback; traceback.print_exc()
            iteration+=1

    # ── FINAL OUTPUT: Best molecule FIRST ────────────────────────────────────
    final=best_first_output(all_scored,nmols)
    mean=sum(all_scored[m][0] for m in final if m in all_scored)/max(1,len(final))
    write_out(final, f"final_mean:{mean:.4f}")
    log(f"Done: {len(final)} molecules, mean score={mean:.4f}")

    db.close()
    log(f"Total time: {time.time()-t0:.0f}s")

if __name__=='__main__':
    main()
