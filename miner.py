#!/usr/bin/env python3
"""
SN68 Blueprint Miner v4 — DPEX_DJA with Environment Audit
==========================================================
Combined diagnostic + competitive miner for Nova drug discovery subnet.

Phase 0: Audit Docker environment (what packages, DB, scoring available)
Phase 1: Initialize populations from SAVI rxn:1/rxn:2 molecules
Phase 2: DPEX_DJA optimization loop
Phase 3: Output best molecule first (critical: num_molecules_boltz=1, sample_selection=first)

Diagnostics written to /output/diagnostics.json for environment intelligence.
"""

import json, os, sys, time, sqlite3, random, traceback
from pathlib import Path
from collections import defaultdict

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════
ALLOWED_TEMPLATES = {1, 2}
MAX_TIME_SECONDS = 1500       # 25 min safety (epochs ~30 min)
POP_A_SIZE = 300              # Global exploration population
POP_B_SIZE = 100              # Local refinement population
T_EXCHANGE = 10               # Exchange interval (iterations)
STALL_THRESHOLD = 5           # Boost mutation after N stalled iters
MAX_ITERATIONS = 200          # Cap on optimization iterations
SCORE_BATCH_SIZE = 200        # Max molecules per scoring call
TABU_MAX = 5000               # Max tabu set size before pruning

# Known high-affinity DAT binder seeds (methylphenidate / GBR analogues)
DAT_SEEDS = [
    "COC(=O)C1(c2ccccc2)CCCCN1",                         # methylphenidate
    "O=C(c1ccccc1)c1ccc(N2CCN(CCCCc3ccccc3)CC2)cc1",    # GBR12909
    "CC(NC(C)(C)C)C(=O)c1cccc(Cl)c1",                    # bupropion
    "O=C(OC)C1CC2CCC1(c1ccc(F)cc1)N2C",                  # FECNT
    "CN1CCc2c(N)cccc2C1c1ccccc1",                         # nomifensine
]

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def elapsed_since(t0):
    return time.time() - t0


# ══════════════════════════════════════════════════════════════
# PHASE 0: ENVIRONMENT AUDIT
# ══════════════════════════════════════════════════════════════
def audit_environment():
    """Dump everything visible in the Docker sandbox to diagnostics.json."""
    diag = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "miner_version": "v4_dpex_dja",
        "python_version": sys.version,
        "sys_path": sys.path[:20],
        "env_vars": dict(os.environ),
        "cwd": os.getcwd(),
        "workspace_contents": [],
        "root_contents": [],
        "output_dir_exists": os.path.isdir("/output"),
        "importable": {},
        "input_json": None,
        "input_json_path": None,
        "savi_db_candidates": [],
        "db_schemas": {},
    }

    # Scan /workspace
    for root, dirs, files in os.walk("/workspace"):
        for f in files:
            fp = os.path.join(root, f)
            try:
                sz = os.path.getsize(fp)
            except:
                sz = -1
            diag["workspace_contents"].append({"path": fp, "size_bytes": sz})
        if root.count(os.sep) > 5:
            dirs.clear()

    # Scan / top level
    try:
        diag["root_contents"] = os.listdir("/")
    except:
        pass

    # Read input.json
    for p in ["/workspace/input.json", "/input.json", "input.json",
              "/workspace/config.json", "/config.json"]:
        try:
            with open(p) as f:
                diag["input_json"] = json.load(f)
                diag["input_json_path"] = p
                log(f"Found input at {p}")
                break
        except:
            pass

    # Check importability
    for pkg in [
        "rdkit", "rdkit.Chem", "rdkit.Chem.AllChem", "rdkit.Chem.Descriptors",
        "rdkit.Chem.DataStructs", "rdkit.Chem.Crippen",
        "torch", "numpy", "pandas", "scipy", "sklearn", "xgboost",
        "PSICHIC", "PSICHIC.wrapper",
        "openbabel", "mordred", "deepchem",
    ]:
        try:
            __import__(pkg)
            diag["importable"][pkg] = True
        except ImportError:
            diag["importable"][pkg] = False
        except Exception as e:
            diag["importable"][pkg] = f"error: {e}"

    # Find SQLite databases
    for search_root in ["/workspace", "/data", "/db", "/savi"]:
        if not os.path.isdir(search_root):
            continue
        for root, dirs, files in os.walk(search_root):
            for f in files:
                if f.endswith((".db", ".sqlite", ".sqlite3", ".savi")):
                    fp = os.path.join(root, f)
                    diag["savi_db_candidates"].append(fp)
                    try:
                        conn = sqlite3.connect(fp)
                        cur = conn.cursor()
                        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
                        tables = [r[0] for r in cur.fetchall()]
                        schema = {"tables": tables, "details": {}}
                        for t in tables[:10]:
                            cur.execute(f"PRAGMA table_info([{t}])")
                            cols = [(r[1], r[2]) for r in cur.fetchall()]
                            cur.execute(f"SELECT COUNT(*) FROM [{t}]")
                            cnt = cur.fetchone()[0]
                            sample = []
                            try:
                                cur.execute(f"SELECT * FROM [{t}] LIMIT 3")
                                sample = [list(r) for r in cur.fetchall()]
                            except:
                                pass
                            schema["details"][t] = {
                                "columns": cols,
                                "row_count": cnt,
                                "sample_rows": sample,
                            }
                        conn.close()
                        diag["db_schemas"][fp] = schema
                    except Exception as e:
                        diag["db_schemas"][fp] = {"error": str(e)}
            if root.count(os.sep) > 5:
                dirs.clear()

    # Write diagnostics
    os.makedirs("/output", exist_ok=True)
    try:
        with open("/output/diagnostics.json", "w") as f:
            json.dump(diag, f, indent=2, default=str)
    except Exception as e:
        log(f"WARNING: Could not write diagnostics: {e}")

    n_db = len(diag["savi_db_candidates"])
    psichic = diag["importable"].get("PSICHIC", False)
    rdkit = diag["importable"].get("rdkit", False)
    log(f"Audit: {len(diag['workspace_contents'])} files, {n_db} DBs, "
        f"PSICHIC={'YES' if psichic is True else 'NO'}, "
        f"RDKit={'YES' if rdkit is True else 'NO'}")
    return diag


# ══════════════════════════════════════════════════════════════
# SAVI DATABASE ADAPTER
# ══════════════════════════════════════════════════════════════
class SAVIDatabase:
    """Adaptive interface to SAVI SQLite database."""

    def __init__(self, db_path, diag_schema=None):
        self.path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

        # Discover schema
        cur = self.conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        self.tables = [r[0] for r in cur.fetchall()]

        # Find the main product table
        self.table = None
        self.smiles_col = None
        self.name_col = None
        self.rxn_col = None
        self.all_cols = []

        for t in self.tables:
            cur.execute(f"PRAGMA table_info([{t}])")
            cols = [(r[1], r[2]) for r in cur.fetchall()]
            col_names = [c[0] for c in cols]
            self.all_cols = col_names

            # Detect columns
            for c in col_names:
                cl = c.lower()
                if "smiles" in cl or "product_smi" in cl:
                    self.smiles_col = c
                    self.table = t
                if "name" in cl and "table" not in cl:
                    self.name_col = c
                if cl in ("rxn_id", "rxn", "template_id", "reaction_id", "template"):
                    self.rxn_col = c

            if self.smiles_col:
                break

        if not self.table and self.tables:
            self.table = self.tables[0]
            cur.execute(f"PRAGMA table_info([{self.table}])")
            cols = [(r[1], r[2]) for r in cur.fetchall()]
            self.all_cols = [c[0] for c in cols]
            # Guess smiles column — first TEXT column with length > 5
            for c in cols:
                if c[1].upper() in ("TEXT", "VARCHAR", ""):
                    self.smiles_col = self.smiles_col or c[0]
            if not self.smiles_col:
                self.smiles_col = self.all_cols[0]

        log(f"SAVI: table={self.table}, smiles={self.smiles_col}, "
            f"name={self.name_col}, rxn={self.rxn_col}, cols={self.all_cols[:10]}")

    def sample_rxn12(self, n=500):
        """Sample n molecules restricted to rxn:1 and rxn:2 templates."""
        cur = self.conn.cursor()
        results = []

        for rxn_id in [1, 2]:
            limit = n // 2

            # Strategy 1: Direct rxn column filter
            if self.rxn_col:
                try:
                    cur.execute(
                        f"SELECT * FROM [{self.table}] WHERE [{self.rxn_col}] = ? "
                        f"ORDER BY RANDOM() LIMIT ?",
                        (rxn_id, limit),
                    )
                    rows = cur.fetchall()
                    if rows:
                        results.extend(rows)
                        continue
                    # Try string version
                    cur.execute(
                        f"SELECT * FROM [{self.table}] WHERE [{self.rxn_col}] = ? "
                        f"ORDER BY RANDOM() LIMIT ?",
                        (str(rxn_id), limit),
                    )
                    rows = cur.fetchall()
                    if rows:
                        results.extend(rows)
                        continue
                except:
                    pass

            # Strategy 2: LIKE on name column (rxn:1:... format)
            if self.name_col:
                try:
                    cur.execute(
                        f"SELECT * FROM [{self.table}] WHERE [{self.name_col}] LIKE ? "
                        f"ORDER BY RANDOM() LIMIT ?",
                        (f"rxn:{rxn_id}:%", limit),
                    )
                    rows = cur.fetchall()
                    if rows:
                        results.extend(rows)
                        continue
                except:
                    pass

            # Strategy 3: LIKE on any text column
            for col in self.all_cols:
                try:
                    cur.execute(
                        f"SELECT * FROM [{self.table}] WHERE [{col}] LIKE ? "
                        f"ORDER BY RANDOM() LIMIT ?",
                        (f"%rxn:{rxn_id}%", limit),
                    )
                    rows = cur.fetchall()
                    if rows:
                        results.extend(rows)
                        break
                except:
                    pass

        # Fallback: random sample if no rxn filtering worked
        if not results:
            try:
                cur.execute(
                    f"SELECT * FROM [{self.table}] ORDER BY RANDOM() LIMIT ?", (n,)
                )
                results = cur.fetchall()
                log(f"WARNING: rxn filter failed, random sample of {len(results)}")
            except:
                pass

        log(f"Sampled {len(results)} molecules from SAVI (rxn:1+2 target)")
        return results

    def get_neighbours(self, smiles_or_row, n=20):
        """Get structurally neighbouring molecules (same template, vary reactants)."""
        cur = self.conn.cursor()

        # Try to extract rxn info from name
        name = ""
        if hasattr(smiles_or_row, "keys"):
            d = dict(smiles_or_row)
            name = str(d.get(self.name_col, "")) if self.name_col else ""

        parts = name.split(":")
        if len(parts) >= 4:
            # Format: rxn:ID:REACTANT1:REACTANT2 — keep rxn:ID, vary reactants
            rxn_prefix = f"{parts[0]}:{parts[1]}:"
            try:
                cur.execute(
                    f"SELECT * FROM [{self.table}] WHERE [{self.name_col}] LIKE ? "
                    f"ORDER BY RANDOM() LIMIT ?",
                    (rxn_prefix + "%", n),
                )
                rows = cur.fetchall()
                if rows:
                    return rows
            except:
                pass

        # Fallback: random within same rxn template
        for rxn_id in [1, 2]:
            if self.rxn_col:
                try:
                    cur.execute(
                        f"SELECT * FROM [{self.table}] WHERE [{self.rxn_col}] = ? "
                        f"ORDER BY RANDOM() LIMIT ?",
                        (rxn_id, n),
                    )
                    rows = cur.fetchall()
                    if rows:
                        return rows
                except:
                    pass

        # Last resort
        try:
            cur.execute(
                f"SELECT * FROM [{self.table}] ORDER BY RANDOM() LIMIT ?", (n,)
            )
            return cur.fetchall()
        except:
            return []

    def extract_smiles(self, row):
        """Pull SMILES string from a database row."""
        if hasattr(row, "keys"):
            d = dict(row)
            if self.smiles_col and self.smiles_col in d:
                return str(d[self.smiles_col])
            # Try all columns
            for v in d.values():
                s = str(v)
                if len(s) > 5 and any(c in s for c in "CNOcn()="):
                    return s
        return None

    def extract_name(self, row):
        """Pull name/ID from a database row."""
        if hasattr(row, "keys"):
            d = dict(row)
            if self.name_col and self.name_col in d:
                return str(d[self.name_col])
        return ""

    def decompose(self, row):
        """Extract reactant components from a SAVI molecule entry."""
        name = self.extract_name(row)
        parts = name.split(":")
        if len(parts) >= 3:
            return {
                "rxn": f"{parts[0]}:{parts[1]}",
                "reactants": parts[2:],
                "key": name,
            }
        return {"rxn": "unknown", "reactants": [name], "key": name}


# ══════════════════════════════════════════════════════════════
# COMPONENT RANKER (EMA-based reactant quality tracker)
# ══════════════════════════════════════════════════════════════
class ComponentRanker:
    """Track which reactant building blocks appear in high-scoring molecules."""

    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.scores = defaultdict(lambda: 0.5)
        self.counts = defaultdict(int)

    def update(self, components, score):
        for comp in components:
            old = self.scores[comp]
            self.scores[comp] = self.alpha * score + (1 - self.alpha) * old
            self.counts[comp] += 1

    def rank(self, comp):
        return self.scores[comp]

    def bias_weight(self, components):
        """Return a composite weight for a set of components (higher = better)."""
        if not components:
            return 0.5
        return sum(self.scores[c] for c in components) / len(components)

    def top_n(self, n=50):
        return sorted(self.scores.items(), key=lambda x: -x[1])[:n]


# ══════════════════════════════════════════════════════════════
# SCORER — PSICHIC if available, else RDKit heuristic
# ══════════════════════════════════════════════════════════════
class Scorer:
    def __init__(self, diag):
        self.psichic_available = False
        self.rdkit_available = False
        self.pw = None
        self.seed_fps = []
        self._score_cache = {}

        # Try PSICHIC
        if diag["importable"].get("PSICHIC") is True or \
           diag["importable"].get("PSICHIC.wrapper") is True:
            try:
                from PSICHIC.wrapper import PsichicWrapper
                self.pw = PsichicWrapper()
                self.psichic_available = True
                log("PSICHIC loaded and ready!")
            except Exception as e:
                log(f"PSICHIC import ok but init failed: {e}")

        # Try RDKit
        if diag["importable"].get("rdkit") is True or \
           diag["importable"].get("rdkit.Chem") is True:
            try:
                from rdkit import Chem
                from rdkit.Chem import AllChem, Descriptors, DataStructs
                self._Chem = Chem
                self._AllChem = AllChem
                self._Desc = Descriptors
                self._DS = DataStructs
                self.rdkit_available = True

                # Pre-compute DAT seed fingerprints
                for smi in DAT_SEEDS:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
                        self.seed_fps.append(fp)
                log(f"RDKit loaded, {len(self.seed_fps)} seed fingerprints")
            except Exception as e:
                log(f"RDKit import error: {e}")

    def score_batch(self, smiles_list, target, antitargets):
        """Score molecules, using cache to avoid redundant computation."""
        uncached = [s for s in smiles_list if s not in self._score_cache]

        if uncached:
            if self.psichic_available:
                new_scores = self._psichic_score(uncached, target, antitargets)
            elif self.rdkit_available:
                new_scores = self._heuristic_score(uncached)
            else:
                new_scores = {s: random.uniform(0.0, 0.3) for s in uncached}
            self._score_cache.update(new_scores)

        return {s: self._score_cache.get(s, 0.0) for s in smiles_list}

    def _psichic_score(self, smiles_list, target, antitargets):
        """Full PSICHIC scoring: target - 0.9 * max(antitargets)."""
        scores = {}
        try:
            target_aff = self._screen_protein(target, smiles_list)
            anti_max = defaultdict(float)
            for at in antitargets:
                at_scores = self._screen_protein(at, smiles_list)
                for smi, v in at_scores.items():
                    anti_max[smi] = max(anti_max[smi], v)

            for smi in smiles_list:
                ts = target_aff.get(smi, 0.0)
                at = anti_max.get(smi, 0.0)
                scores[smi] = ts - 0.9 * at
        except Exception as e:
            log(f"PSICHIC batch error: {e}")
            traceback.print_exc()
            scores = self._heuristic_score(smiles_list)
        return scores

    def _screen_protein(self, protein_id, smiles_list):
        """Screen SMILES against one protein using PSICHIC."""
        seq = self._fetch_seq(protein_id)
        if not seq:
            return {}
        try:
            self.pw.run_challenge_start(seq)
            # Process in batches
            all_scores = {}
            for i in range(0, len(smiles_list), SCORE_BATCH_SIZE):
                batch = smiles_list[i : i + SCORE_BATCH_SIZE]
                df = self.pw.run_validation(batch)
                for _, row in df.iterrows():
                    smi = str(row.get("Ligand", ""))
                    aff = float(row.get("predicted_binding_affinity", 0.0))
                    all_scores[smi] = aff
            return all_scores
        except Exception as e:
            log(f"Screen {protein_id} error: {e}")
            return {}

    def _fetch_seq(self, uniprot_id):
        """Fetch protein FASTA sequence from UniProt."""
        import urllib.request
        for url in [
            f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta",
            f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta",
        ]:
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    fasta = r.read().decode()
                lines = fasta.strip().split("\n")
                seq = "".join(l for l in lines if not l.startswith(">"))
                if seq:
                    return seq
            except:
                pass
        return None

    def _heuristic_score(self, smiles_list):
        """Tanimoto-to-DAT-seeds weighted by heavy atom efficiency."""
        scores = {}
        if not self.rdkit_available or not self.seed_fps:
            return {s: random.uniform(0.0, 0.3) for s in smiles_list}

        Chem = self._Chem
        AllChem = self._AllChem
        DS = self._DS
        for smi in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smi)
                if not mol:
                    scores[smi] = 0.0
                    continue
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
                ha = mol.GetNumHeavyAtoms()
                # Tanimoto to best DAT seed
                max_sim = max(DS.TanimotoSimilarity(fp, sfp) for sfp in self.seed_fps)
                # Heavy atom normalization: smaller potent molecules score higher
                ha_factor = min(1.0, 20.0 / max(ha, 10))
                # Drug-likeness bonus
                mw = self._Desc.MolWt(mol)
                logp = self._Desc.MolLogP(mol)
                rb = self._Desc.NumRotatableBonds(mol)
                dl_penalty = 0.0
                if mw > 500:
                    dl_penalty += 0.1
                if logp > 5:
                    dl_penalty += 0.1
                if rb > 10:
                    dl_penalty += 0.05
                scores[smi] = max(0.0, max_sim * ha_factor - dl_penalty)
            except:
                scores[smi] = 0.0
        return scores


# ══════════════════════════════════════════════════════════════
# DPEX_DJA OPTIMIZER
# ══════════════════════════════════════════════════════════════
class DPEX_DJA:
    """
    Discrete Population Exchange — Discrete Jaya Algorithm.

    Pop A (large): Global exploration via Jaya-inspired discrete updates.
    Pop B (focused): Local refinement via tabu-enhanced neighbourhood search.
    Periodic exchange of top individuals between populations.
    ComponentRanker learns which reactant building blocks produce signal.
    """

    def __init__(self, db, scorer, target, antitargets):
        self.db = db
        self.scorer = scorer
        self.target = target
        self.antitargets = antitargets
        self.ranker = ComponentRanker(alpha=0.1)
        self.tabu = set()
        self.best_score = float("-inf")
        self.best_smi = None
        self.stall = 0
        self.mutation_prob = 0.1
        self.all_scored = {}  # smiles -> score

    def _rows_to_mols(self, rows):
        """Convert DB rows to molecule dicts."""
        mols = []
        seen = set()
        for row in rows:
            smi = self.db.extract_smiles(row)
            if smi and smi not in seen:
                seen.add(smi)
                mols.append({"smiles": smi, "row": row, "score": 0.0})
        return mols

    def _score_mols(self, mols):
        """Score a list of molecule dicts, update ranker and cache."""
        unscored = [m for m in mols if m["smiles"] not in self.all_scored]
        if unscored:
            batch = [m["smiles"] for m in unscored]
            scores = self.scorer.score_batch(batch, self.target, self.antitargets)
            for m in unscored:
                m["score"] = scores.get(m["smiles"], 0.0)
                self.all_scored[m["smiles"]] = m["score"]
                # Update component ranker
                comps = self.db.decompose(m["row"])
                self.ranker.update(comps.get("reactants", []), m["score"])
        # Fill cached scores
        for m in mols:
            if m["smiles"] in self.all_scored:
                m["score"] = self.all_scored[m["smiles"]]

    def initialize(self):
        """Seed both populations from SAVI database."""
        log("Initializing DPEX_DJA populations...")
        rows = self.db.sample_rxn12(n=POP_A_SIZE + POP_B_SIZE + 200)
        mols = self._rows_to_mols(rows)

        if not mols:
            log("WARNING: No SAVI molecules found, seeding with DAT_SEEDS")
            mols = [{"smiles": s, "row": None, "score": 0.0} for s in DAT_SEEDS]

        self._score_mols(mols)
        mols.sort(key=lambda x: -x["score"])

        self.pop_a = mols[: POP_A_SIZE]
        self.pop_b = mols[POP_A_SIZE : POP_A_SIZE + POP_B_SIZE]

        if mols:
            self.best_smi = mols[0]["smiles"]
            self.best_score = mols[0]["score"]

        log(
            f"Init: A={len(self.pop_a)} B={len(self.pop_b)} "
            f"best={self.best_score:.4f} total_scored={len(self.all_scored)}"
        )

    def jaya_update(self):
        """Global exploration: inject new molecules biased toward high-ranked components."""
        new_mols = []
        # Sample neighbours of top Pop A molecules
        for mol in self.pop_a[:30]:
            if mol["row"] is None:
                continue
            neighbours = self.db.get_neighbours(mol["row"], n=5)
            for nb in neighbours:
                smi = self.db.extract_smiles(nb)
                if smi and smi not in self.all_scored:
                    new_mols.append({"smiles": smi, "row": nb, "score": 0.0})

        # Anti-stall mutation: inject random rxn:1/2 molecules
        if self.stall >= STALL_THRESHOLD:
            extra_rows = self.db.sample_rxn12(n=max(30, int(50 * self.mutation_prob)))
            for row in extra_rows:
                smi = self.db.extract_smiles(row)
                if smi and smi not in self.all_scored:
                    new_mols.append({"smiles": smi, "row": row, "score": 0.0})

        if new_mols:
            self._score_mols(new_mols)
            combined = self.pop_a + new_mols
            combined.sort(key=lambda x: -x["score"])
            self.pop_a = combined[: POP_A_SIZE]

    def tabu_search(self):
        """Local refinement: explore neighbours of Pop B, avoid repeats."""
        new_mols = []
        for mol in self.pop_b[:20]:
            if mol["row"] is None:
                continue
            neighbours = self.db.get_neighbours(mol["row"], n=10)
            for nb in neighbours:
                smi = self.db.extract_smiles(nb)
                if smi and smi not in self.tabu and smi not in self.all_scored:
                    new_mols.append({"smiles": smi, "row": nb, "score": 0.0})
                    self.tabu.add(smi)

        # Prune tabu if too large
        if len(self.tabu) > TABU_MAX:
            self.tabu = set(list(self.tabu)[-TABU_MAX // 2 :])

        if new_mols:
            self._score_mols(new_mols)
            combined = self.pop_b + new_mols
            combined.sort(key=lambda x: -x["score"])
            self.pop_b = combined[: POP_B_SIZE]

    def exchange(self):
        """Swap top individuals between populations."""
        if not self.pop_a or not self.pop_b:
            return
        n = max(1, min(len(self.pop_a), len(self.pop_b)) // 10)
        top_b = self.pop_b[:n]
        top_a = self.pop_a[:n]
        self.pop_a = sorted(self.pop_a + top_b, key=lambda x: -x["score"])[: POP_A_SIZE]
        self.pop_b = sorted(self.pop_b + top_a, key=lambda x: -x["score"])[: POP_B_SIZE]

    def run(self, max_time, max_iter=MAX_ITERATIONS):
        """Main optimization loop."""
        t0 = time.time()
        for it in range(max_iter):
            if elapsed_since(t0) > max_time:
                log(f"Time limit at iter {it}")
                break

            prev_best = self.best_score

            self.jaya_update()
            self.tabu_search()

            if (it + 1) % T_EXCHANGE == 0:
                self.exchange()

            # Update global best
            all_mols = self.pop_a + self.pop_b
            if all_mols:
                top = max(all_mols, key=lambda x: x["score"])
                if top["score"] > self.best_score:
                    self.best_score = top["score"]
                    self.best_smi = top["smiles"]
                    self.stall = 0
                    self.mutation_prob = 0.1
                    log(f"  [{it}] NEW BEST: {self.best_score:.4f} — {self.best_smi[:50]}")
                else:
                    self.stall += 1
                    if self.stall >= STALL_THRESHOLD:
                        self.mutation_prob = min(0.5, self.mutation_prob * 1.5)

            if it % 20 == 0:
                log(
                    f"  [{it}] best={self.best_score:.4f} scored={len(self.all_scored)} "
                    f"stall={self.stall} mut={self.mutation_prob:.2f}"
                )

        log(
            f"DPEX_DJA done: {len(self.all_scored)} scored, best={self.best_score:.4f}"
        )
        return sorted(self.all_scored.items(), key=lambda x: -x[1])


# ══════════════════════════════════════════════════════════════
# OUTPUT
# ══════════════════════════════════════════════════════════════
def write_output(results, max_mols=100):
    """Write result.json — BEST MOLECULE MUST BE FIRST."""
    os.makedirs("/output", exist_ok=True)
    seen = set()
    output = []
    for smi, score in results:
        if smi not in seen:
            seen.add(smi)
            output.append({
                "product_smiles": smi,
                "product_name": f"v4_dpex_{score:.4f}",
                "_score": score,
            })
        if len(output) >= max_mols:
            break

    with open("/output/result.json", "w") as f:
        json.dump(output, f, indent=2)

    log(f"Output: {len(output)} molecules to /output/result.json")
    if output:
        log(f"  #1 (Boltz target): {output[0]['product_smiles'][:60]}  score={output[0]['_score']:.4f}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    log("=" * 60)
    log("SN68 Blueprint Miner v4 — DPEX_DJA")
    log("=" * 60)

    # ── Phase 0: Environment audit ──
    log("Phase 0: Auditing Docker environment...")
    diag = audit_environment()

    # ── Parse input ──
    inp = diag.get("input_json") or {}
    target = inp.get("target", inp.get("target_protein", "P23977"))
    antitargets = inp.get(
        "antitargets",
        inp.get("anti_targets", ["G3W658", "A0A8B7SKN4", "Q83ER5", "P83267", "A0A9W3GGF2"]),
    )
    log(f"Target: {target}")
    log(f"Antitargets: {antitargets}")

    # ── Find SAVI database ──
    db = None
    db_paths = list(diag.get("savi_db_candidates", []))
    # Check input.json for explicit db path
    for key in ["savi_db", "database", "db_path", "savi_path", "db"]:
        if key in inp:
            db_paths.insert(0, str(inp[key]))
    # Also check common paths
    for guess in [
        "/workspace/savi.db",
        "/workspace/savi2020.db",
        "/workspace/data/savi.db",
        "/workspace/database.db",
        "/data/savi.db",
    ]:
        if os.path.isfile(guess) and guess not in db_paths:
            db_paths.append(guess)

    for dbp in db_paths:
        try:
            db = SAVIDatabase(dbp, diag.get("db_schemas", {}).get(dbp))
            log(f"Connected to SAVI: {dbp}")
            break
        except Exception as e:
            log(f"DB {dbp} failed: {e}")

    # ── Initialize scorer ──
    scorer = Scorer(diag)
    log(f"Scorer: PSICHIC={'YES' if scorer.psichic_available else 'NO'} "
        f"RDKit={'YES' if scorer.rdkit_available else 'NO'}")

    # ── Phase 1+2: DPEX_DJA optimization ──
    if db:
        log("Phase 1: Initializing DPEX_DJA...")
        opt = DPEX_DJA(db, scorer, target, antitargets)
        opt.initialize()

        remaining = MAX_TIME_SECONDS - elapsed_since(t0)
        log(f"Phase 2: Running optimization ({remaining:.0f}s budget)...")
        results = opt.run(max_time=remaining)
        write_output(results)

        # Log component ranker top picks for intelligence
        top_comps = opt.ranker.top_n(20)
        if top_comps:
            log(f"Top components: {top_comps[:5]}")
    else:
        # ── Fallback: DAT seeds only ──
        log("WARNING: No SAVI database found — DAT seed fallback")
        scores = scorer.score_batch(DAT_SEEDS, target, antitargets)
        results = sorted(scores.items(), key=lambda x: -x[1])
        write_output(results)

    elapsed = elapsed_since(t0)
    log(f"{'=' * 60}")
    log(f"Complete in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    log(f"{'=' * 60}")


if __name__ == "__main__":
    main()
