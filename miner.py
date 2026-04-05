#!/usr/bin/env python3
"""
miner.py — DPEX_DJA Blueprint Miner v8.5 for SN68 Nova
Dual-Population Discrete Jaya Algorithm with ComponentRanker
8.5/10 Competitive Build — Hardened for unknown sandbox environments

Features:
  - Correct nova_ph2 imports (actual sandbox API)
  - PSICHIC scoring via PsichicWrapper (GPU-accelerated)
  - Robust scoring fallback chain (PSICHIC → RDKit heuristic → random)
  - Diversity-preserving selection (Tanimoto-based)
  - Multi-objective greedy output (score + diversity + size)
  - Pop A = 500 (matches UID 3)
  - Small molecule bias (target 20-30 heavy atoms for normalization advantage)
  - Anti-plateau with adaptive mutation
  - Cross-reaction awareness from input.json
  - [F] Pre-computed reactant quality prior (QED + heavy atom preference)
  - [F] Biased population initialization (70/30 biased/random)
  - [J] Adaptive time allocation (explore → balanced → exploit phases)
  - [K] Elite archive with novelty search (Tanimoto distance gating)
  - [L] Reaction-aware diversification (pool analysis + small-mol bias)
  - [H] XGBoost/GBR surrogate model (pre-screen 2000→100 candidates per iteration)
  - [I] Cross-epoch memory (warm-start ComponentRanker from previous runs)

Sandbox environment:
  - Reads /workspace/input.json (config + challenge)
  - Writes /output/result.json (molecule names list)
  - nova_ph2 package available (PSICHIC, combinatorial_db, utils)
  - GPU available, no network access
  - RDKit 2024.9.4, PyTorch 2.7.1, ESM2 pre-cached
"""

from __future__ import annotations

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import sys
import json
import time
import math
import random
import sqlite3
import hashlib
import traceback
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Set
from collections import defaultdict

# ---------------------------------------------------------------------------
# Logging — fallback if bittensor not available
# ---------------------------------------------------------------------------
class _FallbackLog:
    @staticmethod
    def info(msg): print(f"[INFO] {msg}", flush=True)
    @staticmethod
    def warning(msg): print(f"[WARN] {msg}", flush=True)
    @staticmethod
    def error(msg): print(f"[ERROR] {msg}", flush=True)
    @staticmethod
    def debug(msg): print(f"[DEBUG] {msg}", flush=True)

try:
    import bittensor as bt
    log = bt.logging
except ImportError:
    log = _FallbackLog()

# ---------------------------------------------------------------------------
# RDKit imports
# ---------------------------------------------------------------------------
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem, DataStructs
    from rdkit import RDLogger
    RDLogger.logger().setLevel(RDLogger.ERROR)
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    log.warning("RDKit not available")

# ---------------------------------------------------------------------------
# PSICHIC scoring via nova_ph2 (correct sandbox import)
# ---------------------------------------------------------------------------
HAS_PSICHIC = False
psichic_wrapper = None
psichic_failed_permanently = False  # Set True after first GPU/init failure

try:
    from nova_ph2.PSICHIC.wrapper import PsichicWrapper
    HAS_PSICHIC = True
    log.info("PSICHIC wrapper imported successfully")
except ImportError as e:
    log.warning(f"PSICHIC wrapper not available: {e}")

# ---------------------------------------------------------------------------
# Combinatorial DB imports via nova_ph2
# ---------------------------------------------------------------------------
HAS_COMB_DB = False
try:
    from nova_ph2.combinatorial_db.reactions import (
        get_smiles_from_reaction,
    )
    HAS_COMB_DB = True
    log.info("Combinatorial DB imported successfully")
except ImportError as e:
    log.warning(f"Combinatorial DB not available: {e}")

# ---------------------------------------------------------------------------
# Feature H: XGBoost/sklearn imports for surrogate model
# ---------------------------------------------------------------------------
HAS_XGBOOST = False
HAS_SKLEARN = False
try:
    from sklearn.ensemble import GradientBoostingRegressor
    HAS_SKLEARN = True
    log.info("sklearn GradientBoostingRegressor available")
except ImportError:
    log.warning("sklearn not available — trying xgboost fallback")

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
    log.info("xgboost available")
except ImportError:
    log.warning("xgboost not available")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INPUT_PATH = os.environ.get("WORKDIR", "/workspace") + "/input.json"
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "result.json")
SCRATCH_DIR = "/tmp/dpex_dja"

POP_A_SIZE = 500       # Global exploration (matches UID 3)
POP_B_SIZE = 150       # Local exploitation
STALL_THRESHOLD = 3    # Faster anti-plateau trigger
EMA_ALPHA = 0.3        # ComponentRanker smoothing
EXCHANGE_INTERVAL = 2  # More frequent population exchange
TIME_RESERVE_SEC = 45  # Safety margin before deadline
SCORE_BATCH_SIZE = 50  # Molecules per PSICHIC batch

# Target heavy atom range (normalization advantage)
IDEAL_MIN_HA = 20
IDEAL_MAX_HA = 35

# Feature J: Adaptive time allocation phases (fraction of usable budget)
PHASE_EXPLORE_END = 0.40   # First 40%: broad exploration
PHASE_BALANCED_END = 0.70  # Next 30%: balanced
# Final 30%: exploitation

# Feature K: Elite archive
ELITE_ARCHIVE_SIZE = 200      # Max elite archive members
NOVELTY_THRESHOLD = 0.3       # Min Tanimoto distance to enter archive
ELITE_INJECT_RATE = 0.05      # Fraction of pop replaced with archive members each iter

# Feature L: Reaction-aware pool analysis
SMALL_MOL_SAMPLE_SIZE = 200   # Reactants to sample for size analysis

# Feature H: XGBoost surrogate model
SURROGATE_MIN_SAMPLES = 30      # Min training pairs before model is usable
SURROGATE_PRESCREEN_N = 2000    # Candidates to generate for prescreening
SURROGATE_PRESCREEN_TOP = 100   # Top candidates to keep after prescreening

# Feature I: Cross-epoch memory
EPOCH_MEMORY_DIR = SCRATCH_DIR  # Persist epoch state in scratch


# ===========================================================================
# ComponentRanker — tracks EMA quality of individual reactants
# ===========================================================================
class ComponentRanker:
    """EMA-based quality tracker for individual reactants."""

    def __init__(self, alpha: float = EMA_ALPHA):
        self.alpha = alpha
        self.ema: Dict[int, float] = {}
        self.count: Dict[int, int] = {}

    def seed_prior(self, mol_id: int, prior_score: float):
        """Seed a reactant with a pre-computed quality prior.
        Only sets initial value; subsequent updates via update() will blend in."""
        if mol_id not in self.ema:
            self.ema[mol_id] = prior_score
            self.count[mol_id] = 1

    def update(self, mol_ids: List[int], score: float):
        for mid in mol_ids:
            if mid not in self.ema:
                self.ema[mid] = score
                self.count[mid] = 1
            else:
                self.ema[mid] = self.alpha * score + (1 - self.alpha) * self.ema[mid]
                self.count[mid] += 1

    def get_weights(self, mol_ids: List[int]) -> List[float]:
        if not mol_ids:
            return []
        scores = [self.ema.get(mid, 0.0) for mid in mol_ids]
        min_s = min(scores) if scores else 0
        return [s - min_s + 1e-6 for s in scores]

    def top_k(self, k: int = 50) -> List[int]:
        return sorted(self.ema.keys(), key=lambda x: self.ema[x], reverse=True)[:k]


# ===========================================================================
# Feature K: Elite Archive — best-ever molecules with novelty gating
# ===========================================================================
class EliteArchive:
    """Maintains a novelty-gated archive of the best molecules ever seen."""

    def __init__(self, max_size: int = ELITE_ARCHIVE_SIZE,
                 novelty_threshold: float = NOVELTY_THRESHOLD):
        self.max_size = max_size
        self.novelty_threshold = novelty_threshold
        self.members: List[Individual] = []
        self._fingerprints: List = []  # parallel to members

    def try_add(self, ind: Individual) -> bool:
        """Add individual if it passes novelty and quality gates.
        Returns True if added."""
        if not ind.valid or ind.score <= float('-inf'):
            return False

        # Novelty gate: must be sufficiently different from all archive members
        if HAS_RDKIT and ind.fingerprint is not None and self._fingerprints:
            for fp in self._fingerprints:
                if fp is not None:
                    sim = tanimoto_similarity(ind.fingerprint, fp)
                    if sim > (1.0 - self.novelty_threshold):
                        # Too similar — only replace if score is much better
                        idx = self._fingerprints.index(fp)
                        if ind.score > self.members[idx].score * 1.1:
                            self.members[idx] = ind
                            self._fingerprints[idx] = ind.fingerprint
                            return True
                        return False

        # Archive not full — add directly
        if len(self.members) < self.max_size:
            self.members.append(ind)
            self._fingerprints.append(ind.fingerprint)
            return True

        # Archive full — replace worst if new is better
        worst_idx = min(range(len(self.members)),
                        key=lambda i: self.members[i].score)
        if ind.score > self.members[worst_idx].score:
            self.members[worst_idx] = ind
            self._fingerprints[worst_idx] = ind.fingerprint
            return True

        return False

    def get_top(self, n: int) -> List[Individual]:
        """Return top-n members by score."""
        return sorted(self.members, key=lambda x: x.score, reverse=True)[:n]

    def get_random_elite(self, n: int) -> List[Individual]:
        """Return n random archive members for injection into populations."""
        if not self.members:
            return []
        return random.sample(self.members, min(n, len(self.members)))

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def best_score(self) -> float:
        if not self.members:
            return float('-inf')
        return max(m.score for m in self.members)

    @property
    def avg_score(self) -> float:
        if not self.members:
            return 0.0
        return sum(m.score for m in self.members) / len(self.members)


# ===========================================================================
# Feature H: XGBoost Surrogate Model
# ===========================================================================
class SurrogateModel:
    """Lightweight surrogate model for pre-screening candidates.

    Trained on (molecular descriptors → PSICHIC score) pairs.
    Allows 50-100x speedup by pre-screening thousands of candidates
    before expensive real PSICHIC scoring.
    """

    def __init__(self):
        self.model = None
        self.features_list: List[Tuple[List[float], float]] = []
        self._trained = False

    def _extract_descriptors(self, smiles: str) -> Optional[List[float]]:
        """Extract RDKit molecular descriptors from SMILES.
        Returns [MW, LogP, HBD, HBA, TPSA, RotBonds, AromaticRings, HeavyAtoms] or None."""
        if not HAS_RDKIT or not smiles:
            return None
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            mw = float(Descriptors.MolWt(mol))
            logp = float(Descriptors.MolLogP(mol))
            hbd = float(Descriptors.NumHDonors(mol))
            hba = float(Descriptors.NumHAcceptors(mol))
            tpsa = float(Descriptors.TPSA(mol))
            rotbonds = float(Descriptors.NumRotatableBonds(mol))
            aromatic = float(Descriptors.NumAromaticRings(mol))
            heavy = float(mol.GetNumHeavyAtoms())
            return [mw, logp, hbd, hba, tpsa, rotbonds, aromatic, heavy]
        except Exception as e:
            log.debug(f"Descriptor extraction failed for {smiles[:40]}: {e}")
            return None

    def add_training_data(self, smiles_list: List[str], scores: List[float]):
        """Add (SMILES, score) pairs to training set."""
        if not smiles_list or not scores or len(smiles_list) != len(scores):
            return
        try:
            for smiles, score in zip(smiles_list, scores):
                desc = self._extract_descriptors(smiles)
                if desc is not None:
                    self.features_list.append((desc, score))
        except Exception as e:
            log.warning(f"Error adding training data to surrogate: {e}")

    def train(self):
        """Fit the surrogate model if we have enough training data."""
        if len(self.features_list) < SURROGATE_MIN_SAMPLES:
            return
        if self._trained:
            return  # Already trained
        try:
            X = [f[0] for f in self.features_list]
            y = [f[1] for f in self.features_list]

            # Use sklearn GradientBoostingRegressor as primary, xgboost as fallback
            if HAS_SKLEARN:
                self.model = GradientBoostingRegressor(
                    n_estimators=50,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
                log.info(f"Training GradientBoostingRegressor on {len(X)} samples")
            elif HAS_XGBOOST:
                self.model = XGBRegressor(
                    n_estimators=50,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42,
                    verbosity=0
                )
                log.info(f"Training XGBRegressor on {len(X)} samples")
            else:
                log.warning("Neither sklearn nor xgboost available — surrogate disabled")
                return

            self.model.fit(X, y)
            self._trained = True
            log.info(f"Surrogate model trained on {len(X)} samples")
        except Exception as e:
            log.warning(f"Surrogate training failed: {e}")
            self.model = None
            self._trained = False

    def predict(self, smiles_list: List[str]) -> List[float]:
        """Predict scores for a list of SMILES.
        Returns list of predictions, or empty list if not ready."""
        if not self._trained or self.model is None:
            return []
        if not smiles_list:
            return []
        try:
            descriptors = []
            for smiles in smiles_list:
                desc = self._extract_descriptors(smiles)
                if desc is not None:
                    descriptors.append(desc)
                else:
                    descriptors.append([0.0] * 8)  # Fallback neutral vector
            if not descriptors:
                return []
            preds = self.model.predict(descriptors)
            return [float(p) for p in preds]
        except Exception as e:
            log.warning(f"Surrogate prediction failed: {e}")
            return []

    @property
    def is_ready(self) -> bool:
        """True if model is trained and ready to use."""
        return self._trained and self.model is not None


# ===========================================================================
# Feature I: Cross-Epoch Memory
# ===========================================================================
class EpochMemory:
    """Persist ComponentRanker state across validator rounds.

    Allows warm-starting from previous runs on the same reaction.
    Uses JSON files in SCRATCH_DIR for persistence.
    """

    def __init__(self):
        try:
            os.makedirs(SCRATCH_DIR, exist_ok=True)
        except Exception as e:
            log.warning(f"Failed to create SCRATCH_DIR: {e}")

    def _get_filename(self, rxn_id: int) -> str:
        """Get the JSON filename for a reaction ID."""
        return os.path.join(SCRATCH_DIR, f"epoch_memory_rxn{rxn_id}.json")

    def save(self, rxn_id: int, ranker: ComponentRanker, top_molecules: List[str]):
        """Save ranker state and top molecules to disk."""
        if ranker is None:
            return
        try:
            data = {
                "rxn_id": rxn_id,
                "ranker_ema": ranker.ema,
                "ranker_count": ranker.count,
                "top_molecules": top_molecules,
            }
            filepath = self._get_filename(rxn_id)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            log.info(f"Epoch memory saved to {filepath}")
        except Exception as e:
            log.warning(f"Failed to save epoch memory: {e}")

    def load(self, rxn_id: int) -> Optional[Dict]:
        """Load ranker state for a reaction ID.
        Returns dict with 'ranker_ema', 'ranker_count', 'top_molecules' or None."""
        try:
            filepath = self._get_filename(rxn_id)
            if not os.path.exists(filepath):
                return None
            with open(filepath, 'r') as f:
                data = json.load(f)
            log.info(f"Epoch memory loaded from {filepath}")
            return data
        except Exception as e:
            log.warning(f"Failed to load epoch memory: {e}")
            return None


# ===========================================================================
# SAVI Database Interface
# ===========================================================================
class SAVIDatabase:
    """Interface to the combinatorial molecules.sqlite database.

    Hardened: discovers table names and column names dynamically.
    Does not assume any specific schema — adapts to whatever is in the DB.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)

        # Dynamic schema discovery
        self._discover_schema()
        self.reactions = self._load_reactions()
        self.role_pools: Dict[int, List[Tuple[int, str, int]]] = {}

    def _discover_schema(self):
        """Discover table and column names dynamically."""
        cursor = self.conn.cursor()

        # Find all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cursor.fetchall()]
        log.info(f"DB tables: {tables}")

        # Find reaction table (try common names)
        self.rxn_table = None
        for candidate in ["reactions", "reaction", "rxn", "rxns"]:
            if candidate in tables:
                self.rxn_table = candidate
                break
        if not self.rxn_table:
            # Pick any table with 'rxn' or 'react' in name
            for t in tables:
                if "rxn" in t.lower() or "react" in t.lower():
                    self.rxn_table = t
                    break
        if not self.rxn_table and tables:
            # Last resort: if only 2 tables, one is likely reactions
            if len(tables) <= 3:
                for t in tables:
                    if t.lower() != "molecules" and t.lower() != "sqlite_sequence":
                        self.rxn_table = t
                        break

        # Find molecule table
        self.mol_table = None
        for candidate in ["molecules", "molecule", "mols", "compounds", "reagents"]:
            if candidate in tables:
                self.mol_table = candidate
                break
        if not self.mol_table:
            for t in tables:
                if "mol" in t.lower() or "comp" in t.lower() or "reagent" in t.lower():
                    self.mol_table = t
                    break
        if not self.mol_table and tables:
            for t in tables:
                if t != self.rxn_table and t.lower() != "sqlite_sequence":
                    self.mol_table = t
                    break

        log.info(f"Detected tables: rxn={self.rxn_table}, mol={self.mol_table}")

        # Discover column names for each table
        self.rxn_cols = {}
        if self.rxn_table:
            cursor.execute(f"PRAGMA table_info({self.rxn_table})")
            cols = [r[1] for r in cursor.fetchall()]
            log.info(f"Reaction table columns: {cols}")
            # Map canonical names to actual column names
            col_map = {
                "rxn_id": ["rxn_id", "id", "reaction_id", "rid"],
                "smarts": ["smarts", "reaction_smarts", "rxn_smarts", "template"],
                "roleA": ["roleA", "role_a", "rolea", "role1", "reactant_role_1"],
                "roleB": ["roleB", "role_b", "roleb", "role2", "reactant_role_2"],
                "roleC": ["roleC", "role_c", "rolec", "role3", "reactant_role_3"],
            }
            for canonical, aliases in col_map.items():
                for alias in aliases:
                    if alias in cols:
                        self.rxn_cols[canonical] = alias
                        break
                    # Case-insensitive fallback
                    for c in cols:
                        if c.lower() == alias.lower():
                            self.rxn_cols[canonical] = c
                            break
                    if canonical in self.rxn_cols:
                        break
            log.info(f"Reaction column mapping: {self.rxn_cols}")

        self.mol_cols = {}
        if self.mol_table:
            cursor.execute(f"PRAGMA table_info({self.mol_table})")
            cols = [r[1] for r in cursor.fetchall()]
            log.info(f"Molecule table columns: {cols}")
            col_map = {
                "mol_id": ["mol_id", "id", "molecule_id", "mid", "compound_id"],
                "smiles": ["smiles", "smi", "SMILES", "canonical_smiles"],
                "role_mask": ["role_mask", "role", "roles", "rolemask"],
            }
            for canonical, aliases in col_map.items():
                for alias in aliases:
                    if alias in cols:
                        self.mol_cols[canonical] = alias
                        break
                    for c in cols:
                        if c.lower() == alias.lower():
                            self.mol_cols[canonical] = c
                            break
                    if canonical in self.mol_cols:
                        break
            # If we can't find role_mask, check if the table uses a different structure
            if "role_mask" not in self.mol_cols:
                log.warning(f"No role_mask column found in {cols}. "
                            f"Will try positional fallback.")
            log.info(f"Molecule column mapping: {self.mol_cols}")

    def _load_reactions(self) -> List[Tuple]:
        if not self.rxn_table:
            log.error("No reaction table found!")
            return []
        cursor = self.conn.cursor()
        # Build query from discovered columns
        rxn_id_col = self.rxn_cols.get("rxn_id", "rxn_id")
        smarts_col = self.rxn_cols.get("smarts", "smarts")
        roleA_col = self.rxn_cols.get("roleA", "roleA")
        roleB_col = self.rxn_cols.get("roleB", "roleB")
        roleC_col = self.rxn_cols.get("roleC", "roleC")

        try:
            cursor.execute(
                f"SELECT {rxn_id_col}, {smarts_col}, {roleA_col}, "
                f"{roleB_col}, {roleC_col} FROM {self.rxn_table}"
            )
        except sqlite3.OperationalError as e:
            log.warning(f"Column-based reaction query failed: {e}")
            # Fallback: select all columns and hope for the best
            cursor.execute(f"SELECT * FROM {self.rxn_table}")

        rows = cursor.fetchall()
        log.info(f"Loaded {len(rows)} reactions. Sample: {rows[0] if rows else 'none'}")
        result = []
        for r in rows:
            try:
                result.append((int(r[0]), str(r[1]), int(r[2]), int(r[3]),
                               int(r[4]) if r[4] is not None else None))
            except (ValueError, TypeError, IndexError):
                result.append(r)
        return result

    def get_reaction_by_id(self, rxn_id: int) -> Optional[Tuple]:
        for r in self.reactions:
            try:
                if int(r[0]) == rxn_id:
                    return r
            except (ValueError, TypeError):
                continue
        # If not found by ID, return first reaction as fallback
        if self.reactions:
            log.warning(f"Reaction {rxn_id} not found. Using first reaction: {self.reactions[0][0]}")
            return self.reactions[0]
        return None

    def get_molecules_by_role(self, role_mask: int, limit: int = 0) -> List[Tuple[int, str, int]]:
        if role_mask in self.role_pools:
            return self.role_pools[role_mask]
        if not self.mol_table:
            log.error("No molecule table found!")
            return []

        cursor = self.conn.cursor()
        mol_id_col = self.mol_cols.get("mol_id", "mol_id")
        smiles_col = self.mol_cols.get("smiles", "smiles")
        role_col = self.mol_cols.get("role_mask", "role_mask")

        try:
            query = (f"SELECT {mol_id_col}, {smiles_col}, {role_col} "
                     f"FROM {self.mol_table} WHERE ({role_col} & ?) = ?")
            if limit > 0:
                query += f" LIMIT {limit}"
            cursor.execute(query, (role_mask, role_mask))
        except sqlite3.OperationalError as e:
            log.warning(f"Role-based molecule query failed: {e}")
            # Fallback: try exact match instead of bitwise
            try:
                query = (f"SELECT {mol_id_col}, {smiles_col}, {role_col} "
                         f"FROM {self.mol_table} WHERE {role_col} = ?")
                if limit > 0:
                    query += f" LIMIT {limit}"
                cursor.execute(query, (role_mask,))
            except sqlite3.OperationalError as e2:
                log.warning(f"Exact role query also failed: {e2}")
                # Last resort: grab everything
                try:
                    cursor.execute(f"SELECT * FROM {self.mol_table} LIMIT 10000")
                except Exception:
                    return []

        results = cursor.fetchall()
        # Ensure tuples are (int, str, int) shaped
        cleaned = []
        for r in results:
            try:
                cleaned.append((int(r[0]), str(r[1]), int(r[2]) if len(r) > 2 else role_mask))
            except (ValueError, TypeError, IndexError):
                continue
        self.role_pools[role_mask] = cleaned
        log.info(f"Loaded {len(cleaned)} molecules for role_mask={role_mask}")
        return cleaned

    def get_molecule_by_id(self, mol_id: int) -> Optional[Tuple[int, str, int]]:
        if not self.mol_table:
            return None
        cursor = self.conn.cursor()
        mol_id_col = self.mol_cols.get("mol_id", "mol_id")
        smiles_col = self.mol_cols.get("smiles", "smiles")
        role_col = self.mol_cols.get("role_mask", "role_mask")
        try:
            cursor.execute(
                f"SELECT {mol_id_col}, {smiles_col}, {role_col} "
                f"FROM {self.mol_table} WHERE {mol_id_col} = ?", (mol_id,))
        except sqlite3.OperationalError:
            cursor.execute(f"SELECT * FROM {self.mol_table} WHERE rowid = ?", (mol_id,))
        row = cursor.fetchone()
        return row

    def count_molecules(self) -> int:
        if not self.mol_table:
            return 0
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {self.mol_table}")
        return cursor.fetchone()[0]

    def close(self):
        self.conn.close()


# ===========================================================================
# Molecule Validation & Fingerprinting
# ===========================================================================
def validate_molecule(smiles: str, config: dict) -> bool:
    """Check if molecule meets config constraints."""
    if not smiles or not HAS_RDKIT:
        return bool(smiles)  # If no RDKit, accept if smiles exists
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        ha = mol.GetNumHeavyAtoms()
        if ha < config.get("min_heavy_atoms", 20):
            return False
        rot = Descriptors.NumRotatableBonds(mol)
        if rot < config.get("min_rotatable_bonds", 1):
            return False
        if rot > config.get("max_rotatable_bonds", 10):
            return False
        # Check for banned atoms
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "Se":
                return False
        return True
    except Exception:
        return False


def get_heavy_atom_count(smiles: str) -> int:
    """Get heavy atom count for a SMILES string."""
    if not HAS_RDKIT or not smiles:
        return 25  # default estimate
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol.GetNumHeavyAtoms() if mol else 25
    except Exception:
        return 25


def get_fingerprint(smiles: str):
    """Get Morgan fingerprint for Tanimoto similarity."""
    if not HAS_RDKIT or not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    except Exception:
        return None


def _extract_scores(result, expected_len: int) -> List[float]:
    """Extract numeric scores from PSICHIC result — handles DataFrame, dict, list, array."""
    if result is None:
        return []
    try:
        # pandas DataFrame
        if hasattr(result, 'columns'):
            # Try known column names
            for col_name in ['predicted_binding_affinity', 'binding_affinity', 'score',
                             'affinity', 'prediction', 'pba', 'output']:
                if col_name in result.columns:
                    return result[col_name].tolist()
            # Try first numeric column
            for col in result.columns:
                if result[col].dtype in ['float64', 'float32', 'float16', 'int64', 'int32']:
                    return result[col].tolist()
            # Last resort: first column
            if len(result.columns) > 0:
                return result.iloc[:, 0].tolist()
        # dict with scores
        if isinstance(result, dict):
            for key in ['scores', 'predictions', 'binding_affinity', 'output']:
                if key in result:
                    val = result[key]
                    if isinstance(val, list):
                        return [float(x) for x in val]
                    if hasattr(val, 'tolist'):
                        return val.tolist()
        # raw list or numpy array
        if isinstance(result, (list, tuple)):
            return [float(x) for x in result]
        if hasattr(result, 'tolist'):
            return result.tolist()
    except Exception as e:
        log.warning(f"Score extraction failed: {e}")
    return []


def tanimoto_similarity(fp1, fp2) -> float:
    """Calculate Tanimoto similarity between two fingerprints."""
    if fp1 is None or fp2 is None:
        return 0.0
    try:
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception:
        return 0.0


# ===========================================================================
# Individual — represents a single molecule solution
# ===========================================================================
class Individual:
    __slots__ = ['rxn_id', 'mol_ids', 'name', 'smiles', 'score',
                 'valid', 'heavy_atoms', 'fingerprint', 'normalized_score']

    def __init__(self, rxn_id: int, mol_ids: List[int], name: str = "",
                 smiles: str = "", score: float = float('-inf'),
                 valid: bool = False):
        self.rxn_id = rxn_id
        self.mol_ids = mol_ids
        self.name = name
        self.smiles = smiles
        self.score = score
        self.valid = valid
        self.heavy_atoms = 0
        self.fingerprint = None
        self.normalized_score = float('-inf')

    def compute_properties(self):
        """Compute heavy atoms and fingerprint after SMILES is known."""
        if self.smiles and HAS_RDKIT:
            self.heavy_atoms = get_heavy_atom_count(self.smiles)
            self.fingerprint = get_fingerprint(self.smiles)

    def __repr__(self):
        return f"Ind({self.name[:30]}, score={self.score:.4f})"


# ===========================================================================
# DPEX_DJA Optimizer v5
# ===========================================================================
class DPEX_DJA:
    """Dual-Population Discrete Jaya Algorithm v5."""

    def __init__(self, db: SAVIDatabase, config: dict, rxn_id: int):
        self.db = db
        self.config = config
        self.rxn_id = rxn_id
        self.ranker = ComponentRanker()

        rxn_info = db.get_reaction_by_id(rxn_id)
        if not rxn_info:
            # Hardened: never crash — try first available reaction
            if db.reactions:
                rxn_info = db.reactions[0]
                self.rxn_id = int(rxn_info[0])
                log.warning(f"Reaction {rxn_id} not found, falling back to {self.rxn_id}")
            else:
                raise ValueError(f"No reactions in database at all")
        self.smarts = rxn_info[1]
        self.roleA = int(rxn_info[2])
        self.roleB = int(rxn_info[3])
        self.roleC = int(rxn_info[4]) if rxn_info[4] and int(rxn_info[4]) != 0 else None
        self.is_three_component = self.roleC is not None

        # Pre-fetch reactant pools
        self.pool_A = db.get_molecules_by_role(self.roleA)
        self.pool_B_react = db.get_molecules_by_role(self.roleB)
        self.pool_C = db.get_molecules_by_role(self.roleC) if self.is_three_component else []

        # Index pools by mol_id
        self.idx_A = {m[0]: m for m in self.pool_A}
        self.idx_B = {m[0]: m for m in self.pool_B_react}
        self.idx_C = {m[0]: m for m in self.pool_C} if self.is_three_component else {}

        # Populations
        self.pop_a: List[Individual] = []
        self.pop_b: List[Individual] = []

        # Best ever
        self.best_ever: Optional[Individual] = None

        # Stall counter
        self.stall_count = 0
        self.last_best_score = float('-inf')

        # Tabu set
        self.tabu_set: Set[tuple] = set()

        # Feature K: Elite archive
        self.elite_archive = EliteArchive()

        # Feature H: Surrogate model for pre-screening
        self.surrogate = SurrogateModel()

        # Feature I: Cross-epoch memory
        self.epoch_memory = EpochMemory()

        # Feature J: Phase tracking
        self.current_phase = "explore"  # explore, balanced, exploit

        # Feature L: Reaction-aware pool preferences
        self.small_mol_reactants_A: List[int] = []
        self.small_mol_reactants_B: List[int] = []

        log.info(
            f"DPEX_DJA v8.5: rxn={rxn_id}, "
            f"poolA={len(self.pool_A)}, poolB={len(self.pool_B_react)}, "
            f"poolC={len(self.pool_C)}, 3-comp={self.is_three_component}"
        )

        # Feature I: Try to warm-start from epoch memory
        try:
            mem_data = self.epoch_memory.load(rxn_id)
            if mem_data:
                if 'ranker_ema' in mem_data and 'ranker_count' in mem_data:
                    self.ranker.ema = mem_data['ranker_ema']
                    self.ranker.count = mem_data['ranker_count']
                    log.info(f"Warm-started ranker with {len(self.ranker.ema)} previously ranked molecules")
        except Exception as e:
            log.warning(f"Could not warm-start from epoch memory: {e}")

        # Feature F: Pre-compute reactant quality priors
        self._compute_reactant_prior()

        # Feature L: Analyze pools for small-molecule bias
        self._analyze_reaction_pools()

    # ------------------------------------------------------------------
    # Feature F: Pre-Computed Reactant Quality Prior
    # ------------------------------------------------------------------
    def _compute_reactant_prior(self):
        """Score every reactant with RDKit druglikeness proxy and seed
        the ComponentRanker so biased selection is smart from iteration 0.

        Uses QED (Quantitative Estimate of Druglikeness) when available,
        falling back to a Lipinski-inspired heuristic.  Also applies a
        heavy-atom preference bonus (20-30 HA sweet spot for the
        normalization advantage the validator uses).
        """
        if not HAS_RDKIT:
            log.info("No RDKit — skipping reactant quality prior")
            return

        # Try importing QED
        try:
            from rdkit.Chem import QED as QED_module
            has_qed = True
        except ImportError:
            has_qed = False

        scored = 0
        all_pools = [
            ("A", self.pool_A),
            ("B", self.pool_B_react),
        ]
        if self.is_three_component:
            all_pools.append(("C", self.pool_C))

        for pool_name, pool in all_pools:
            for mol_id, smiles, role_mask in pool:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        continue

                    ha = mol.GetNumHeavyAtoms()

                    # Base quality score
                    if has_qed:
                        qed_score = QED_module.qed(mol)  # 0-1 range
                        base = qed_score * 3.0  # Scale to ~0-3
                    else:
                        # Lipinski heuristic fallback
                        mw = Descriptors.MolWt(mol)
                        logp = Descriptors.MolLogP(mol)
                        hbd = Descriptors.NumHDonors(mol)
                        hba = Descriptors.NumHAcceptors(mol)
                        base = 0.0
                        if mw < 500:
                            base += 0.8
                        if logp < 5:
                            base += 0.8
                        if hbd <= 5:
                            base += 0.4
                        if hba <= 10:
                            base += 0.4
                        # Ring count — drug molecules usually have rings
                        if Descriptors.RingCount(mol) >= 1:
                            base += 0.3

                    # Heavy-atom preference bonus (normalization advantage)
                    # Validator divides affinity by heavy atoms, so smaller
                    # molecules with decent affinity score higher
                    ha_bonus = 0.0
                    if 20 <= ha <= 25:
                        ha_bonus = 0.5  # Sweet spot
                    elif 25 < ha <= 30:
                        ha_bonus = 0.3
                    elif 30 < ha <= 35:
                        ha_bonus = 0.1

                    prior = base + ha_bonus
                    self.ranker.seed_prior(mol_id, prior)
                    scored += 1
                except Exception:
                    continue

        log.info(f"Reactant quality prior: scored {scored} reactants "
                 f"(QED={'yes' if has_qed else 'no'})")

    # ------------------------------------------------------------------
    # Feature L: Reaction-Aware Pool Analysis
    # ------------------------------------------------------------------
    def _analyze_reaction_pools(self):
        """Identify reactants that tend to produce smaller molecules.
        Smaller products score higher under heavy-atom normalization."""
        if not HAS_RDKIT:
            return

        t0 = time.time()
        # Sample reactants and measure heavy atom counts
        ha_by_mol: Dict[int, List[int]] = defaultdict(list)

        sample_A = random.sample(self.pool_A, min(SMALL_MOL_SAMPLE_SIZE, len(self.pool_A)))
        sample_B = random.sample(self.pool_B_react, min(SMALL_MOL_SAMPLE_SIZE, len(self.pool_B_react)))

        for molA in sample_A:
            try:
                mol = Chem.MolFromSmiles(molA[1])
                if mol:
                    ha_by_mol[molA[0]].append(mol.GetNumHeavyAtoms())
            except Exception:
                pass

        for molB in sample_B:
            try:
                mol = Chem.MolFromSmiles(molB[1])
                if mol:
                    ha_by_mol[molB[0]].append(mol.GetNumHeavyAtoms())
            except Exception:
                pass

        # Identify small-molecule reactants (HA < 18 = likely to form 20-30 HA products)
        for mol_id, ha_list in ha_by_mol.items():
            avg_ha = sum(ha_list) / len(ha_list)
            if avg_ha <= 18:
                if mol_id in self.idx_A:
                    self.small_mol_reactants_A.append(mol_id)
                elif mol_id in self.idx_B:
                    self.small_mol_reactants_B.append(mol_id)

        elapsed = time.time() - t0
        log.info(f"Reaction pool analysis: {len(self.small_mol_reactants_A)} small-A, "
                 f"{len(self.small_mol_reactants_B)} small-B ({elapsed:.1f}s)")

    def _small_mol_individual(self) -> Individual:
        """Generate individual biased toward small-molecule reactants (Feature L)."""
        for _ in range(20):
            # Use small-mol reactants if available, else fall back to biased
            if self.small_mol_reactants_A:
                mol_id_A = random.choice(self.small_mol_reactants_A)
                molA = self.idx_A.get(mol_id_A)
            else:
                molA = self._biased_select(self.pool_A)

            if self.small_mol_reactants_B:
                mol_id_B = random.choice(self.small_mol_reactants_B)
                molB = self.idx_B.get(mol_id_B)
            else:
                molB = self._biased_select(self.pool_B_react)

            molC = self._biased_select(self.pool_C) if self.is_three_component else None

            if molA and molB:
                ind = self._make_individual(molA, molB, molC)
                if ind.valid and ind.heavy_atoms <= 30:
                    return ind
        return self._biased_individual()

    # ------------------------------------------------------------------
    # Feature J: Update phase based on elapsed time
    # ------------------------------------------------------------------
    def update_phase(self, elapsed_fraction: float):
        """Update optimization phase based on time elapsed."""
        old_phase = self.current_phase
        if elapsed_fraction < PHASE_EXPLORE_END:
            self.current_phase = "explore"
        elif elapsed_fraction < PHASE_BALANCED_END:
            self.current_phase = "balanced"
        else:
            self.current_phase = "exploit"

        if self.current_phase != old_phase:
            log.info(f"PHASE SHIFT: {old_phase} → {self.current_phase} "
                     f"(elapsed={elapsed_fraction:.0%})")

    def get_phase_params(self) -> dict:
        """Return population sizes and mutation rates for current phase."""
        if self.current_phase == "explore":
            return {
                "pop_a_target": POP_A_SIZE,
                "pop_b_target": POP_B_SIZE,
                "mutation_base": 0.20,      # Higher base mutation
                "exchange_interval": 3,      # Less frequent exchange
                "small_mol_fraction": 0.15,  # Some small-mol injection
            }
        elif self.current_phase == "balanced":
            return {
                "pop_a_target": int(POP_A_SIZE * 0.8),
                "pop_b_target": int(POP_B_SIZE * 1.3),
                "mutation_base": 0.10,
                "exchange_interval": 2,
                "small_mol_fraction": 0.20,
            }
        else:  # exploit
            return {
                "pop_a_target": int(POP_A_SIZE * 0.5),
                "pop_b_target": int(POP_B_SIZE * 2.0),
                "mutation_base": 0.05,       # Low mutation — refine
                "exchange_interval": 1,      # Very frequent exchange
                "small_mol_fraction": 0.30,  # Heavy small-mol bias
            }

    # ------------------------------------------------------------------
    # Feature K: Archive injection
    # ------------------------------------------------------------------
    def inject_elites(self):
        """Inject elite archive members into populations to prevent drift."""
        if self.elite_archive.size == 0:
            return

        n_inject_a = max(1, int(len(self.pop_a) * ELITE_INJECT_RATE))
        n_inject_b = max(1, int(len(self.pop_b) * ELITE_INJECT_RATE))

        elites_a = self.elite_archive.get_random_elite(n_inject_a)
        elites_b = self.elite_archive.get_random_elite(n_inject_b)

        # Replace worst members in each population
        if elites_a and self.pop_a:
            scored_a = [(i, ind.score) for i, ind in enumerate(self.pop_a)]
            scored_a.sort(key=lambda x: x[1])
            for j, elite in enumerate(elites_a):
                if j < len(scored_a):
                    idx = scored_a[j][0]
                    if elite.score > self.pop_a[idx].score:
                        self.pop_a[idx] = elite

        if elites_b and self.pop_b:
            scored_b = [(i, ind.score) for i, ind in enumerate(self.pop_b)]
            scored_b.sort(key=lambda x: x[1])
            for j, elite in enumerate(elites_b):
                if j < len(scored_b):
                    idx = scored_b[j][0]
                    if elite.score > self.pop_b[idx].score:
                        self.pop_b[idx] = elite

    def _make_individual(self, molA: Tuple, molB: Tuple,
                         molC: Optional[Tuple] = None) -> Individual:
        idA, _, _ = molA
        idB, _, _ = molB
        if self.is_three_component and molC:
            idC, _, _ = molC
            name = f"rxn:{self.rxn_id}:{idA}:{idB}:{idC}"
            mol_ids = [idA, idB, idC]
        else:
            name = f"rxn:{self.rxn_id}:{idA}:{idB}"
            mol_ids = [idA, idB]

        # Get SMILES using nova_ph2 API if available, else manual
        smiles = None
        if HAS_COMB_DB:
            try:
                smiles = get_smiles_from_reaction(name)
            except Exception:
                pass

        if smiles is None:
            # Fallback: try manual reaction
            smiles = self._manual_react(molA, molB, molC)

        valid = smiles is not None and validate_molecule(smiles, self.config)
        ind = Individual(self.rxn_id, mol_ids, name, smiles or "", float('-inf'), valid)
        if valid and smiles:
            ind.compute_properties()
        return ind

    def _manual_react(self, molA, molB, molC=None) -> Optional[str]:
        """Manual SMARTS reaction as fallback."""
        if not HAS_RDKIT:
            return None
        try:
            _, smiA, _ = molA
            _, smiB, _ = molB
            rxn = AllChem.ReactionFromSmarts(self.smarts)
            if rxn is None:
                return None
            mA = Chem.MolFromSmiles(smiA)
            mB = Chem.MolFromSmiles(smiB)
            if mA is None or mB is None:
                return None
            products = rxn.RunReactants((mA, mB))
            if products and products[0]:
                return Chem.MolToSmiles(products[0][0])
        except Exception:
            pass
        return None

    def _random_individual(self) -> Individual:
        for _ in range(30):
            molA = random.choice(self.pool_A)
            molB = random.choice(self.pool_B_react)
            molC = random.choice(self.pool_C) if self.is_three_component else None
            ind = self._make_individual(molA, molB, molC)
            if ind.valid:
                return ind
        molA = random.choice(self.pool_A)
        molB = random.choice(self.pool_B_react)
        molC = random.choice(self.pool_C) if self.is_three_component else None
        return self._make_individual(molA, molB, molC)

    def _biased_individual(self) -> Individual:
        """Generate individual biased toward high-quality reactants."""
        for _ in range(30):
            # Biased selection for each pool
            molA = self._biased_select(self.pool_A)
            molB = self._biased_select(self.pool_B_react)
            molC = self._biased_select(self.pool_C) if self.is_three_component else None
            ind = self._make_individual(molA, molB, molC)
            if ind.valid:
                return ind
        return self._random_individual()

    def _biased_select(self, pool: List[Tuple]) -> Tuple:
        """Select from pool biased by ComponentRanker weights."""
        if not pool:
            raise ValueError("Empty pool")
        pool_ids = [m[0] for m in pool]
        weights = self.ranker.get_weights(pool_ids)
        if weights and sum(weights) > 0:
            return random.choices(pool, weights=weights, k=1)[0]
        return random.choice(pool)

    def initialize_populations(self, n_a: int = POP_A_SIZE, n_b: int = POP_B_SIZE):
        log.info(f"Initializing Pop A ({n_a}) and Pop B ({n_b})...")
        self.pop_a = []
        self.pop_b = []

        # Build Pop A — 70% biased (use reactant prior), 30% random (diversity)
        n_biased_a = int(n_a * 0.7)
        for _ in range(n_biased_a * 2):
            if len(self.pop_a) >= n_biased_a:
                break
            ind = self._biased_individual()
            if ind.valid:
                self.pop_a.append(ind)
        for _ in range((n_a - len(self.pop_a)) * 3):
            if len(self.pop_a) >= n_a:
                break
            ind = self._random_individual()
            if ind.valid:
                self.pop_a.append(ind)

        # Build Pop B — 100% biased (exploitation from best reactants)
        for _ in range(n_b * 3):
            if len(self.pop_b) >= n_b:
                break
            ind = self._biased_individual()
            if ind.valid:
                self.pop_b.append(ind)

        log.info(f"Populations ready: A={len(self.pop_a)}, B={len(self.pop_b)}")

    def _jaya_mutate(self, current: Individual, best: Individual,
                     worst: Individual) -> Individual:
        """Discrete Jaya update rule."""
        new_mol_ids = []
        pools = [self.pool_A, self.pool_B_react]
        idxs = [self.idx_A, self.idx_B]
        if self.is_three_component:
            pools.append(self.pool_C)
            idxs.append(self.idx_C)

        for slot_idx in range(len(current.mol_ids)):
            curr_id = current.mol_ids[slot_idx]
            best_id = best.mol_ids[slot_idx]
            worst_id = worst.mol_ids[slot_idx]
            pool = pools[slot_idx]

            r1 = random.random()
            r2 = random.random()

            if r1 > 0.5 and best_id in idxs[slot_idx]:
                new_mol_ids.append(best_id)
            elif r2 > 0.5 and worst_id != curr_id:
                candidates = [m for m in pool if m[0] != worst_id]
                if candidates:
                    choice = self._biased_select(candidates)
                    new_mol_ids.append(choice[0])
                else:
                    new_mol_ids.append(curr_id)
            else:
                new_mol_ids.append(curr_id)

        molA = self.idx_A.get(new_mol_ids[0])
        molB = self.idx_B.get(new_mol_ids[1])
        molC = self.idx_C.get(new_mol_ids[2]) if self.is_three_component and len(new_mol_ids) > 2 else None

        if molA and molB:
            return self._make_individual(molA, molB, molC)
        return self._random_individual()

    def _neighbourhood_search(self, ind: Individual) -> Individual:
        """Tabu neighbourhood search for Pop B."""
        pools = [self.pool_A, self.pool_B_react]
        if self.is_three_component:
            pools.append(self.pool_C)

        slot = random.randint(0, len(ind.mol_ids) - 1)
        new_reactant = self._biased_select(pools[slot])

        new_mol_ids = list(ind.mol_ids)
        new_mol_ids[slot] = new_reactant[0]

        key = tuple(new_mol_ids)
        if key in self.tabu_set:
            return self._biased_individual()
        self.tabu_set.add(key)

        # Keep tabu set manageable
        if len(self.tabu_set) > 50000:
            self.tabu_set = set(list(self.tabu_set)[-25000:])

        molA = self.idx_A.get(new_mol_ids[0])
        molB = self.idx_B.get(new_mol_ids[1])
        molC = self.idx_C.get(new_mol_ids[2]) if self.is_three_component and len(new_mol_ids) > 2 else None
        if molA and molB:
            return self._make_individual(molA, molB, molC)
        return self._biased_individual()

    def score_batch_psichic(self, individuals: List[Individual],
                            target_seq: str, antitarget_seqs: List[str],
                            antitarget_weight: float) -> List[Individual]:
        """Score molecules using PSICHIC wrapper."""
        global psichic_wrapper, psichic_failed_permanently, HAS_PSICHIC

        # Skip PSICHIC entirely if it failed before (e.g. no GPU)
        if psichic_failed_permanently:
            return self._fallback_score(individuals)

        valid_inds = [ind for ind in individuals if ind.valid and ind.smiles]
        if not valid_inds:
            return individuals

        smiles_list = [ind.smiles for ind in valid_inds]

        try:
            if psichic_wrapper is None:
                psichic_wrapper = PsichicWrapper()
                log.info("PsichicWrapper initialized")
                # Log available methods for debugging
                methods = [m for m in dir(psichic_wrapper) if not m.startswith("_")]
                log.info(f"PsichicWrapper methods: {methods}")

            # Score against target — try multiple API patterns
            try:
                psichic_wrapper.initialize_model(target_seq)
            except TypeError:
                # Maybe it takes (sequence, device) or just sequence differently
                try:
                    psichic_wrapper.initialize_model(target_seq, "cuda:0")
                except Exception:
                    pass  # Hope score_molecules still works
            except AttributeError:
                # Maybe it's init_model or load_model
                for method_name in ["init_model", "load_model", "set_target", "setup"]:
                    if hasattr(psichic_wrapper, method_name):
                        getattr(psichic_wrapper, method_name)(target_seq)
                        break

            target_df = psichic_wrapper.score_molecules(smiles_list)

            # Extract target scores — handle any DataFrame/dict/list return type
            target_scores = _extract_scores(target_df, len(smiles_list))

            if not target_scores:
                log.warning("No target scores extracted from PSICHIC — trying alternative extraction")
                # Try as raw list/numpy array
                if hasattr(target_df, 'values'):
                    try:
                        target_scores = target_df.values.flatten().tolist()
                    except Exception:
                        pass
                elif isinstance(target_df, (list, tuple)):
                    target_scores = [float(x) for x in target_df]

            if not target_scores:
                log.warning("Still no target scores — falling back")
                return self._fallback_score(individuals)

            # Score against antitargets
            antitarget_scores_all = []
            for aseq in antitarget_seqs:
                try:
                    try:
                        psichic_wrapper.initialize_model(aseq)
                    except Exception:
                        for method_name in ["init_model", "load_model", "set_target"]:
                            if hasattr(psichic_wrapper, method_name):
                                getattr(psichic_wrapper, method_name)(aseq)
                                break
                    anti_df = psichic_wrapper.score_molecules(smiles_list)
                    anti_scores = _extract_scores(anti_df, len(smiles_list))
                    if anti_scores:
                        antitarget_scores_all.append(anti_scores)
                except Exception as e:
                    log.warning(f"Antitarget scoring failed: {e}")

            # Compute final scores
            for i, ind in enumerate(valid_inds):
                if i < len(target_scores):
                    t_score = target_scores[i]
                    a_score = 0.0
                    if antitarget_scores_all:
                        a_scores = [a[i] for a in antitarget_scores_all if i < len(a)]
                        a_score = sum(a_scores) / len(a_scores) if a_scores else 0.0
                    ind.score = t_score - antitarget_weight * a_score

                    # Normalized score (what the validator actually uses)
                    if ind.heavy_atoms > 0:
                        ind.normalized_score = ind.score / ind.heavy_atoms
                    else:
                        ind.normalized_score = ind.score

                    # Update ComponentRanker
                    self.ranker.update(ind.mol_ids, ind.score)

            # Feature H: Feed successful scores to surrogate model for training
            try:
                self.surrogate.add_training_data(smiles_list, target_scores)
                self.surrogate.train()
            except Exception as e:
                log.debug(f"Surrogate training error (non-critical): {e}")

            log.info(f"PSICHIC scored {len(valid_inds)} molecules")
            return individuals

        except Exception as e:
            log.error(f"PSICHIC scoring failed: {e}")
            traceback.print_exc()
            # Disable PSICHIC permanently to avoid wasting time retrying
            psichic_failed_permanently = True
            HAS_PSICHIC = False
            log.warning("PSICHIC disabled for remainder of run — using RDKit heuristic fallback")
            return self._fallback_score(individuals)

    def _fallback_score(self, individuals: List[Individual]) -> List[Individual]:
        """Fallback scoring using RDKit heuristics when PSICHIC fails."""
        log.info("Using RDKit heuristic fallback scoring")
        for ind in individuals:
            if not ind.valid or not ind.smiles or not HAS_RDKIT:
                continue
            try:
                mol = Chem.MolFromSmiles(ind.smiles)
                if mol is None:
                    continue
                # Heuristic: drug-likeness proxy
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                hbd = Descriptors.NumHDonors(mol)
                hba = Descriptors.NumHAcceptors(mol)
                ha = mol.GetNumHeavyAtoms()

                # Lipinski-inspired score (higher = more drug-like)
                score = 0.0
                if mw < 500: score += 1.0
                if logp < 5: score += 1.0
                if hbd <= 5: score += 0.5
                if hba <= 10: score += 0.5
                # Prefer smaller molecules (normalization advantage)
                if 20 <= ha <= 30: score += 1.0
                elif ha <= 35: score += 0.5

                ind.score = score
                ind.normalized_score = score / ha if ha > 0 else 0
                self.ranker.update(ind.mol_ids, ind.score)
            except Exception:
                pass
        return individuals

    def evolve_pop_a(self) -> List[Individual]:
        """Jaya update on Population A with phase-aware mutation."""
        if not self.pop_a:
            return []

        scored = [ind for ind in self.pop_a if ind.score > float('-inf')]
        if len(scored) < 2:
            return self.pop_a

        best = max(scored, key=lambda x: x.score)
        worst = min(scored, key=lambda x: x.score)

        # Phase-aware mutation rate (Feature J)
        phase_params = self.get_phase_params()
        base_mutation = phase_params["mutation_base"]
        stall_boost = min(0.6, 0.15 * self.stall_count) if self.stall_count >= STALL_THRESHOLD else 0.0
        mutation_rate = min(0.8, base_mutation + stall_boost)
        small_mol_rate = phase_params["small_mol_fraction"]

        new_pop = []
        for ind in self.pop_a:
            r = random.random()
            if r < small_mol_rate:
                # Feature L: Small molecule injection
                candidate = self._small_mol_individual()
            elif r < small_mol_rate + mutation_rate:
                candidate = self._biased_individual()
            else:
                candidate = self._jaya_mutate(ind, best, worst)
            if candidate.valid:
                new_pop.append(candidate)
            else:
                new_pop.append(ind)

        # Feature J: Resize population toward phase target
        target_size = phase_params["pop_a_target"]
        if len(new_pop) > target_size:
            # Keep the best, drop worst
            new_pop.sort(key=lambda x: x.score, reverse=True)
            new_pop = new_pop[:target_size]
        elif len(new_pop) < target_size:
            # Fill with biased individuals
            while len(new_pop) < target_size:
                ind = self._biased_individual()
                if ind.valid:
                    new_pop.append(ind)

        return new_pop

    def evolve_pop_b(self) -> List[Individual]:
        """Tabu neighbourhood search on Population B with phase-aware sizing."""
        new_pop = []
        for ind in self.pop_b:
            candidate = self._neighbourhood_search(ind)
            if candidate.valid:
                new_pop.append(candidate)
            else:
                new_pop.append(ind)

        # Feature J: Resize Pop B toward phase target
        phase_params = self.get_phase_params()
        target_size = phase_params["pop_b_target"]
        if len(new_pop) > target_size:
            new_pop.sort(key=lambda x: x.score, reverse=True)
            new_pop = new_pop[:target_size]
        elif len(new_pop) < target_size:
            while len(new_pop) < target_size:
                ind = self._biased_individual()
                if ind.valid:
                    new_pop.append(ind)

        return new_pop

    def exchange_populations(self):
        """Transfer best from each population to the other."""
        if not self.pop_a or not self.pop_b:
            return

        scored_a = [ind for ind in self.pop_a if ind.score > float('-inf')]
        scored_b = [ind for ind in self.pop_b if ind.score > float('-inf')]

        # Transfer top-3 from A to B and vice versa
        n_transfer = 3

        if scored_a:
            top_a = sorted(scored_a, key=lambda x: x.score, reverse=True)[:n_transfer]
            for elite in top_a:
                if self.pop_b:
                    worst_idx = min(range(len(self.pop_b)), key=lambda i: self.pop_b[i].score)
                    if elite.score > self.pop_b[worst_idx].score:
                        self.pop_b[worst_idx] = elite

        if scored_b:
            top_b = sorted(scored_b, key=lambda x: x.score, reverse=True)[:n_transfer]
            for elite in top_b:
                if self.pop_a:
                    worst_idx = min(range(len(self.pop_a)), key=lambda i: self.pop_a[i].score)
                    if elite.score > self.pop_a[worst_idx].score:
                        self.pop_a[worst_idx] = elite

    def update_best(self):
        """Track best-ever, stall counter, and feed elite archive (Feature K)."""
        all_scored = [ind for ind in (self.pop_a + self.pop_b) if ind.score > float('-inf')]
        if not all_scored:
            return

        current_best = max(all_scored, key=lambda x: x.score)
        if self.best_ever is None or current_best.score > self.best_ever.score:
            self.best_ever = current_best
            self.stall_count = 0
            log.info(f"NEW BEST: {current_best.name[:40]} score={current_best.score:.4f}")
        else:
            self.stall_count += 1

        # Feature K: Feed elite archive with top molecules from this iteration
        added = 0
        for ind in sorted(all_scored, key=lambda x: x.score, reverse=True)[:50]:
            if self.elite_archive.try_add(ind):
                added += 1
        if added > 0:
            log.info(f"Elite archive: +{added} (total={self.elite_archive.size}, "
                     f"best={self.elite_archive.best_score:.4f})")

    # ------------------------------------------------------------------
    # Feature H: Surrogate Pre-Screening
    # ------------------------------------------------------------------
    def surrogate_prescreen(self) -> List[Individual]:
        """Generate and pre-screen candidates using trained surrogate model.

        Generates N candidates (e.g., 2000) using existing population operators,
        estimates scores via surrogate, returns top K (e.g., 100) candidates.
        This allows 20-50x speedup by avoiding expensive PSICHIC scoring
        on unpromising candidates.

        Returns: List of pre-screened Individual objects.
        """
        if not self.surrogate.is_ready:
            return []

        try:
            candidates = []
            log.info(f"Generating {SURROGATE_PRESCREEN_N} candidates for surrogate pre-screening...")

            # Generate candidates using standard population operators
            for _ in range(SURROGATE_PRESCREEN_N):
                if random.random() < 0.5:
                    ind = self._biased_individual(is_pop_a=True)
                else:
                    ind = self._small_mol_individual()
                if ind and ind.valid and ind.smiles:
                    candidates.append(ind)

            if not candidates:
                log.warning("No valid candidates generated for surrogate pre-screening")
                return []

            # Extract SMILES
            smiles_list = [ind.smiles for ind in candidates]

            # Get surrogate predictions
            log.info(f"Predicting scores for {len(candidates)} candidates...")
            predictions = self.surrogate.predict(smiles_list)

            if not predictions or len(predictions) != len(candidates):
                log.warning(f"Surrogate prediction count mismatch: {len(predictions)} vs {len(candidates)}")
                return []

            # Assign predicted scores
            for ind, pred in zip(candidates, predictions):
                ind.score = float(pred)
                if ind.heavy_atoms > 0:
                    ind.normalized_score = ind.score / ind.heavy_atoms
                else:
                    ind.normalized_score = ind.score

            # Sort by score and keep top K
            candidates_ranked = sorted(candidates, key=lambda x: x.score, reverse=True)
            top_candidates = candidates_ranked[:SURROGATE_PRESCREEN_TOP]

            log.info(f"Surrogate pre-screening: {len(candidates)} → {len(top_candidates)} "
                    f"(top score={top_candidates[0].score:.4f})")
            return top_candidates

        except Exception as e:
            log.warning(f"Surrogate pre-screening failed: {e}")
            return []

    def get_diverse_top_molecules(self, n: int) -> List[Individual]:
        """
        Multi-objective greedy selection:
        Maximize score while maintaining diversity (Tanimoto distance).
        This feeds the entropy bonus in the validator.
        Includes elite archive members (Feature K).
        """
        # Merge populations + elite archive for final selection
        all_inds = self.pop_a + self.pop_b + self.elite_archive.members
        scored = [ind for ind in all_inds if ind.valid and ind.score > float('-inf')]

        if not scored:
            return [ind for ind in all_inds if ind.valid][:n]

        # Deduplicate by name
        seen = set()
        unique = []
        for ind in sorted(scored, key=lambda x: x.score, reverse=True):
            if ind.name not in seen:
                seen.add(ind.name)
                unique.append(ind)

        if len(unique) <= n:
            return unique

        # Greedy diverse selection
        selected = [unique[0]]  # Start with highest scorer
        remaining = unique[1:]

        while len(selected) < n and remaining:
            best_candidate = None
            best_combined = float('-inf')

            for cand in remaining:
                # Score component (normalized)
                if unique[0].score > unique[-1].score:
                    score_norm = (cand.score - unique[-1].score) / (unique[0].score - unique[-1].score + 1e-10)
                else:
                    score_norm = 1.0

                # Diversity component (min Tanimoto distance to any selected)
                if HAS_RDKIT and cand.fingerprint is not None:
                    min_sim = min(
                        (tanimoto_similarity(cand.fingerprint, s.fingerprint) for s in selected if s.fingerprint is not None),
                        default=0.0
                    )
                    diversity = 1.0 - min_sim
                else:
                    diversity = random.random() * 0.5  # Random diversity if no fingerprints

                # Size preference (closer to 20-25 heavy atoms = better normalized score)
                size_bonus = 0.0
                if cand.heavy_atoms > 0:
                    if IDEAL_MIN_HA <= cand.heavy_atoms <= IDEAL_MAX_HA:
                        size_bonus = 0.1
                    if cand.heavy_atoms <= 25:
                        size_bonus = 0.2

                # Combined objective: 60% score + 30% diversity + 10% size
                combined = 0.6 * score_norm + 0.3 * diversity + 0.1 * size_bonus

                if combined > best_combined:
                    best_combined = combined
                    best_candidate = cand

            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break

        return selected


# ===========================================================================
# Main Entry Point
# ===========================================================================
def load_config() -> dict:
    """Load challenge configuration from input.json.

    Hardened: handles multiple possible schemas — flat dict, nested config/challenge,
    or any combination. Searches for known keys at any nesting depth.
    """
    try:
        with open(INPUT_PATH, "r") as f:
            d = json.load(f)
    except FileNotFoundError:
        log.error(f"input.json not found at {INPUT_PATH}")
        # Try alternative paths
        for alt in ["/workspace/input.json", "/app/input.json", "/input.json",
                    os.path.join(os.getcwd(), "input.json")]:
            if alt != INPUT_PATH and os.path.isfile(alt):
                log.info(f"Found input at alternative path: {alt}")
                with open(alt, "r") as f:
                    d = json.load(f)
                break
        else:
            log.error("No input.json found anywhere — using empty config")
            d = {}
    except json.JSONDecodeError as e:
        log.error(f"input.json is not valid JSON: {e}")
        d = {}

    # Log all top-level keys for debugging
    log.info(f"input.json top-level keys: {list(d.keys()) if isinstance(d, dict) else type(d)}")

    # Flexible merge: try config/challenge nesting, then flat dict
    config = {}
    if isinstance(d, dict):
        # Merge nested sections first
        for section_key in ["config", "challenge", "params", "parameters", "settings"]:
            section = d.get(section_key, {})
            if isinstance(section, dict):
                config.update(section)
        # Then overlay any top-level keys (flat format takes precedence)
        for k, v in d.items():
            if not isinstance(v, dict):  # Skip nested sections already merged
                config[k] = v
            elif k not in ["config", "challenge", "params", "parameters", "settings"]:
                config[k] = v  # Keep unknown nested dicts as-is

    # Normalize common key variations
    _key_aliases = {
        "target_sequences": ["target_sequences", "targets", "target_sequence",
                             "protein_sequences", "sequences"],
        "antitarget_sequences": ["antitarget_sequences", "antitargets",
                                  "anti_target_sequences", "off_targets"],
        "antitarget_weight": ["antitarget_weight", "anti_target_weight",
                               "antitarget_penalty", "selectivity_weight"],
        "allowed_reaction": ["allowed_reaction", "reaction", "reaction_id",
                              "rxn", "rxn_id"],
        "num_molecules": ["num_molecules", "n_molecules", "molecule_count",
                          "num_results", "top_k"],
        "time_budget_sec": ["time_budget_sec", "time_budget", "budget",
                             "timeout", "time_limit"],
        "min_heavy_atoms": ["min_heavy_atoms", "min_ha", "min_heavy_atom_count"],
    }
    for canonical, aliases in _key_aliases.items():
        if canonical not in config:
            for alias in aliases:
                if alias in config and alias != canonical:
                    config[canonical] = config[alias]
                    log.info(f"Config alias: {alias} -> {canonical}")
                    break

    # Ensure target_sequences is always a list
    ts = config.get("target_sequences", [])
    if isinstance(ts, str):
        config["target_sequences"] = [ts]
    ats = config.get("antitarget_sequences", [])
    if isinstance(ats, str):
        config["antitarget_sequences"] = [ats]

    # Normalize allowed_reaction — could be "rxn:1", "1", 1, or {"id": 1}
    ar = config.get("allowed_reaction", None)
    if isinstance(ar, int):
        config["allowed_reaction"] = f"rxn:{ar}"
    elif isinstance(ar, dict):
        rid = ar.get("id", ar.get("rxn_id", ar.get("reaction_id", None)))
        config["allowed_reaction"] = f"rxn:{rid}" if rid is not None else None

    log.info(f"Config: targets={len(config.get('target_sequences', []))}, "
             f"antitargets={len(config.get('antitarget_sequences', []))}, "
             f"num_molecules={config.get('num_molecules', 100)}, "
             f"allowed_reaction={config.get('allowed_reaction', 'none')}, "
             f"min_ha={config.get('min_heavy_atoms', 20)}, "
             f"antitarget_weight={config.get('antitarget_weight', 0.9)}")
    log.info(f"Full config keys: {list(config.keys())}")
    return config


def find_db() -> str:
    """Locate molecules.sqlite in the sandbox.

    Hardened: searches many paths, multiple Python versions, and any .sqlite file
    that looks like a molecule database.
    """
    # Try many possible install paths across Python versions
    candidates = []
    for pyver in ["3.12", "3.11", "3.10", "3.9", "3.13"]:
        for prefix in ["/usr/local/lib", "/usr/lib"]:
            candidates.append(
                f"{prefix}/python{pyver}/site-packages/nova_ph2/combinatorial_db/molecules.sqlite"
            )
            candidates.append(
                f"{prefix}/python{pyver}/dist-packages/nova_ph2/combinatorial_db/molecules.sqlite"
            )
    candidates += [
        "/workspace/combinatorial_db/molecules.sqlite",
        "/workspace/molecules.sqlite",
        "/workspace/data/molecules.sqlite",
        "/app/combinatorial_db/molecules.sqlite",
        "/app/molecules.sqlite",
        "/combinatorial_db/molecules.sqlite",
        "/data/molecules.sqlite",
        "/opt/molecules.sqlite",
    ]

    # Also try to find via nova_ph2 package path
    try:
        import nova_ph2
        pkg_dir = os.path.dirname(nova_ph2.__file__)
        candidates.insert(0, os.path.join(pkg_dir, "combinatorial_db", "molecules.sqlite"))
        candidates.insert(1, os.path.join(pkg_dir, "data", "molecules.sqlite"))
        candidates.insert(2, os.path.join(pkg_dir, "molecules.sqlite"))
        log.info(f"nova_ph2 package at: {pkg_dir}")
    except (ImportError, Exception):
        pass

    for c in candidates:
        if os.path.isfile(c):
            log.info(f"Database found: {c}")
            return c

    # Walk search — look for ANY .sqlite file
    log.info("Candidate paths exhausted — walking filesystem for .sqlite files...")
    found_sqlite = []
    for root_dir in ["/usr/local/lib", "/workspace", "/app", "/opt", "/data", "/home"]:
        if not os.path.isdir(root_dir):
            continue
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for f in filenames:
                if f.endswith(".sqlite") or f.endswith(".db"):
                    path = os.path.join(dirpath, f)
                    found_sqlite.append(path)
                    log.info(f"Found SQLite file: {path}")
            if dirpath.count(os.sep) > 6:
                dirnames.clear()

    # Return best match
    for path in found_sqlite:
        if "molecule" in path.lower():
            log.info(f"Selected molecule DB: {path}")
            return path
    if found_sqlite:
        log.warning(f"No 'molecule' DB found, using first SQLite: {found_sqlite[0]}")
        return found_sqlite[0]

    raise FileNotFoundError("Cannot find any .sqlite database file in the sandbox")


def write_output(molecules: List[str]):
    """Write result.json. Hardened: tries multiple output paths."""
    result = {"molecules": molecules}
    written = False
    for out_dir in [OUTPUT_DIR, "/output", "/workspace/output", "/tmp/output"]:
        try:
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "result.json")
            with open(out_path, "w") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            log.info(f"Wrote {len(molecules)} molecules to {out_path}")
            written = True
            if out_dir == OUTPUT_DIR:
                break  # Primary path worked, done
        except Exception as e:
            log.warning(f"Could not write to {out_dir}: {e}")
    if not written:
        log.error("CRITICAL: Could not write result.json to any path!")


def main():
    start_time = time.time()
    log.info("=" * 60)
    log.info("DPEX_DJA Blueprint Miner v8.5 — starting")
    log.info("=" * 60)

    # ===== COMPREHENSIVE ENVIRONMENT DIAGNOSTIC =====
    log.info(f"Python: {sys.version}")
    log.info(f"PSICHIC: {HAS_PSICHIC}, RDKit: {HAS_RDKIT}, CombDB: {HAS_COMB_DB}")
    log.info(f"CUDA env: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    try:
        import torch
        log.info(f"PyTorch: {torch.__version__}, CUDA avail: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        log.info("PyTorch not available")

    # Directory tree of key paths (discover what's actually in the sandbox)
    for scan_dir in ["/workspace", "/output", "/app", "/opt"]:
        if os.path.isdir(scan_dir):
            log.info(f"Directory tree: {scan_dir}")
            for dirpath, dirnames, filenames in os.walk(scan_dir):
                depth = dirpath.replace(scan_dir, "").count(os.sep)
                if depth > 3:
                    dirnames.clear()
                    continue
                indent = "  " * depth
                log.info(f"{indent}{os.path.basename(dirpath)}/")
                for f in filenames[:20]:  # Cap per-dir listing
                    log.info(f"{indent}  {f}")
                if len(filenames) > 20:
                    log.info(f"{indent}  ... and {len(filenames) - 20} more files")

    # Log key environment variables
    for key in sorted(os.environ):
        if any(kw in key.upper() for kw in ["WORK", "OUTPUT", "INPUT", "CUDA", "PATH",
                                              "MODEL", "BUDGET", "TIME", "GPU", "NOVA"]):
            log.info(f"ENV: {key}={os.environ[key]}")

    # Log raw input.json contents for debugging
    try:
        with open(INPUT_PATH, "r") as _dbg_f:
            raw_input = _dbg_f.read()
        log.info(f"Raw input.json ({len(raw_input)} bytes): {raw_input[:2000]}")
    except Exception as e:
        log.warning(f"Could not read raw input.json: {e}")
    # ===== END DIAGNOSTIC =====

    # 1. Load config
    config = load_config()
    time_budget = config.get("time_budget_sec", 900)
    # Use env var if set (runner may override)
    if "TIME_BUDGET" in os.environ:
        time_budget = int(os.environ["TIME_BUDGET"])
    num_molecules = config.get("num_molecules", 100)
    allowed_reaction = config.get("allowed_reaction", None)
    antitarget_weight = config.get("antitarget_weight", 0.9)
    target_seqs = config.get("target_sequences", [])
    antitarget_seqs = config.get("antitarget_sequences", [])

    deadline = start_time + time_budget - TIME_RESERVE_SEC

    # Determine reaction ID
    rxn_id = None
    if allowed_reaction and allowed_reaction.startswith("rxn:"):
        rxn_id = int(allowed_reaction.split(":")[1])

    # If no valid reaction ID from input, discover from database
    if rxn_id is None or rxn_id == 0:
        db_path_temp = find_db()
        conn_temp = sqlite3.connect(f"file:{db_path_temp}?mode=ro&immutable=1", uri=True)
        cur_temp = conn_temp.cursor()
        cur_temp.execute("SELECT rxn_id FROM reactions ORDER BY rxn_id LIMIT 1")
        row = cur_temp.fetchone()
        conn_temp.close()
        if row:
            rxn_id = row[0]
            log.info(f"No valid allowed_reaction; defaulting to rxn_id={rxn_id}")
        else:
            rxn_id = 1  # Last resort default

    log.info(f"Target rxn_id={rxn_id}, budget={time_budget}s, deadline in {deadline - time.time():.0f}s")

    # 2. Find and connect to SAVI database
    db_path = find_db()
    db = SAVIDatabase(db_path)
    log.info(f"Database: {db_path} ({db.count_molecules()} molecules)")
    log.info(f"Reactions: {[r[0] for r in db.reactions]}")

    # 3. Initialize optimizer
    optimizer = DPEX_DJA(db, config, rxn_id)
    optimizer.initialize_populations()

    # 4. Main optimization loop
    iteration = 0
    target_seq = target_seqs[0] if target_seqs else ""
    usable_budget = deadline - start_time  # Total usable time (budget - reserve)

    while time.time() < deadline:
        iteration += 1
        elapsed = time.time() - start_time
        remaining = deadline - time.time()
        elapsed_fraction = elapsed / usable_budget if usable_budget > 0 else 1.0

        # Feature J: Update optimization phase
        optimizer.update_phase(elapsed_fraction)
        phase_params = optimizer.get_phase_params()

        log.info(f"\n--- Iteration {iteration} | phase={optimizer.current_phase} | "
                 f"elapsed={elapsed:.0f}s | remaining={remaining:.0f}s | "
                 f"popA={len(optimizer.pop_a)} popB={len(optimizer.pop_b)} "
                 f"elite={optimizer.elite_archive.size} ---")

        # Score populations
        if HAS_PSICHIC and target_seq and not psichic_failed_permanently:
            # Score Pop A in batches
            log.info("Scoring Pop A with PSICHIC...")
            for batch_start in range(0, len(optimizer.pop_a), SCORE_BATCH_SIZE):
                if time.time() >= deadline:
                    break
                batch = optimizer.pop_a[batch_start:batch_start + SCORE_BATCH_SIZE]
                optimizer.score_batch_psichic(batch, target_seq, antitarget_seqs, antitarget_weight)

            if time.time() >= deadline:
                break

            # Score Pop B
            log.info("Scoring Pop B with PSICHIC...")
            for batch_start in range(0, len(optimizer.pop_b), SCORE_BATCH_SIZE):
                if time.time() >= deadline:
                    break
                batch = optimizer.pop_b[batch_start:batch_start + SCORE_BATCH_SIZE]
                optimizer.score_batch_psichic(batch, target_seq, antitarget_seqs, antitarget_weight)
        else:
            # Fallback scoring
            optimizer._fallback_score(optimizer.pop_a)
            optimizer._fallback_score(optimizer.pop_b)

        if time.time() >= deadline:
            break

        # Update best tracker + elite archive (Feature K)
        optimizer.update_best()

        # Feature K: Inject elite archive members into populations
        if iteration > 1 and optimizer.elite_archive.size > 0:
            optimizer.inject_elites()

        # Write current best to output (continuously updated)
        top = optimizer.get_diverse_top_molecules(num_molecules)
        if top:
            write_output([ind.name for ind in top])
            scored_top = [ind for ind in top if ind.score > float('-inf')]
            if scored_top:
                avg_score = sum(ind.score for ind in scored_top) / len(scored_top)
                log.info(f"Top {len(top)} avg={avg_score:.4f}, best={scored_top[0].score:.4f}")

        # Evolve populations (phase-aware — Feature J)
        log.info(f"Evolving Pop A (Jaya, {optimizer.current_phase})...")
        optimizer.pop_a = optimizer.evolve_pop_a()

        log.info(f"Evolving Pop B (Neighbourhood, {optimizer.current_phase})...")
        optimizer.pop_b = optimizer.evolve_pop_b()

        # Feature H: Surrogate pre-screening (if model is ready)
        if optimizer.surrogate.is_ready and time.time() < deadline:
            try:
                prescreened = optimizer.surrogate_prescreen()
                if prescreened and len(prescreened) > 0:
                    # Replace one population with pre-screened candidates
                    # (or merge them for more diversity)
                    optimizer.pop_a = prescreened
                    log.info(f"Applied surrogate pre-screening: pop_a now has {len(prescreened)} candidates")
            except Exception as e:
                log.warning(f"Surrogate pre-screening error (non-critical): {e}")

        # Population exchange (phase-aware interval)
        exchange_interval = phase_params["exchange_interval"]
        if iteration % exchange_interval == 0:
            log.info("Exchanging populations...")
            optimizer.exchange_populations()

        if time.time() >= deadline:
            break

    # 5. Final output (includes elite archive members — Feature K)
    top = optimizer.get_diverse_top_molecules(num_molecules)
    if top:
        write_output([ind.name for ind in top])
        scored_top = [ind for ind in top if ind.score > float('-inf')]
        if scored_top:
            log.info(f"\nFINAL: {len(top)} molecules, best={scored_top[0].score:.4f}")
        else:
            log.info(f"\nFINAL: {len(top)} molecules (unscored)")
    else:
        # Last resort: dump all valid molecules (pop + archive)
        all_valid = [ind for ind in (optimizer.pop_a + optimizer.pop_b +
                     optimizer.elite_archive.members) if ind.valid]
        if all_valid:
            write_output([ind.name for ind in all_valid[:num_molecules]])
            log.info(f"\nFALLBACK: {min(len(all_valid), num_molecules)} molecules")
        else:
            write_output([])
            log.error("No valid molecules generated!")

    # Summary stats
    log.info(f"Elite archive: {optimizer.elite_archive.size} members, "
             f"best={optimizer.elite_archive.best_score:.4f}, "
             f"avg={optimizer.elite_archive.avg_score:.4f}")
    log.info(f"Iterations: {iteration}, final phase: {optimizer.current_phase}")

    # Feature I: Save epoch memory for next run
    try:
        top_final = optimizer.get_diverse_top_molecules(num_molecules)
        top_names = [ind.name for ind in top_final]
        optimizer.epoch_memory.save(optimizer.rxn_id, optimizer.ranker, top_names)
        log.info(f"Epoch memory saved for reaction {optimizer.rxn_id}")
    except Exception as e:
        log.warning(f"Failed to save epoch memory: {e}")

    db.close()
    elapsed = time.time() - start_time
    log.info(f"DPEX_DJA v8.5 complete in {elapsed:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"Fatal error: {e}")
        traceback.print_exc()

        # ===== CRASH-PROOF OUTPUT =====
        # Even on total failure, ALWAYS write a result.json.
        # Try every possible output path. An empty list is better than no file.
        for out_dir in [OUTPUT_DIR, "/output", "/workspace/output", "/tmp/output", "."]:
            try:
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, "result.json")
                with open(out_path, "w") as f:
                    json.dump({"molecules": []}, f)
                log.info(f"Crash-proof: wrote empty result to {out_path}")
                break
            except Exception:
                continue

        # Also try to generate SOME molecules even from a crashed state
        # by directly querying the database if we can find it
        try:
            db_path = find_db()
            conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
            cur = conn.cursor()
            # Get first reaction
            cur.execute("SELECT * FROM reactions LIMIT 1")
            rxn = cur.fetchone()
            if rxn:
                rxn_id = rxn[0]
                # Get some molecule IDs for each role
                cur.execute("SELECT * FROM molecules LIMIT 200")
                mols = cur.fetchall()
                if mols and len(mols) >= 2:
                    # Generate molecule names by combining random pairs
                    names = []
                    for i in range(min(100, len(mols) - 1)):
                        names.append(f"rxn:{rxn_id}:{mols[i][0]}:{mols[i+1][0]}")
                    for out_dir in [OUTPUT_DIR, "/output", "/workspace/output"]:
                        try:
                            os.makedirs(out_dir, exist_ok=True)
                            out_path = os.path.join(out_dir, "result.json")
                            with open(out_path, "w") as f:
                                json.dump({"molecules": names[:100]}, f)
                            log.info(f"Crash-proof: wrote {len(names[:100])} emergency molecules to {out_path}")
                            break
                        except Exception:
                            continue
            conn.close()
        except Exception as e2:
            log.error(f"Emergency molecule generation also failed: {e2}")
