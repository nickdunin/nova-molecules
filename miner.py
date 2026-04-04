#!/usr/bin/env python3
"""
miner.py — DPEX_DJA Blueprint Miner v5 for SN68 Nova
Dual-Population Discrete Jaya Algorithm with ComponentRanker
7.5/10 Competitive Build

Improvements over v2:
  - Correct nova_ph2 imports (actual sandbox API)
  - PSICHIC scoring via PsichicWrapper (GPU-accelerated)
  - Robust scoring fallback chain (PSICHIC → RDKit heuristic → random)
  - Diversity-preserving selection (Tanimoto-based)
  - Multi-objective greedy output (score + diversity + size)
  - Pop A = 500 (matches UID 3)
  - Small molecule bias (target 20-30 heavy atoms for normalization advantage)
  - Anti-plateau with adaptive mutation
  - Cross-reaction awareness from input.json

Sandbox environment:
  - Reads /workspace/input.json (config + challenge)
  - Writes /output/result.json (molecule names list)
  - nova_ph2 package available (PSICHIC, combinatorial_db, utils)
  - GPU available, no network access
  - RDKit 2024.9.4, PyTorch 2.7.1, ESM2 pre-cached
"""

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


# ===========================================================================
# ComponentRanker — tracks EMA quality of individual reactants
# ===========================================================================
class ComponentRanker:
    """EMA-based quality tracker for individual reactants."""

    def __init__(self, alpha: float = EMA_ALPHA):
        self.alpha = alpha
        self.ema: Dict[int, float] = {}
        self.count: Dict[int, int] = {}

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
# SAVI Database Interface
# ===========================================================================
class SAVIDatabase:
    """Interface to the combinatorial molecules.sqlite database."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
        self.reactions = self._load_reactions()
        self.role_pools: Dict[int, List[Tuple[int, str, int]]] = {}

    def _load_reactions(self) -> List[Tuple]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT rxn_id, smarts, roleA, roleB, roleC FROM reactions")
        rows = cursor.fetchall()
        result = []
        for r in rows:
            try:
                result.append((int(r[0]), r[1], int(r[2]), int(r[3]), int(r[4]) if r[4] is not None else None))
            except (ValueError, TypeError):
                result.append(r)
        return result

    def get_reaction_by_id(self, rxn_id: int) -> Optional[Tuple]:
        for r in self.reactions:
            if r[0] == rxn_id:
                return r
        return None

    def get_molecules_by_role(self, role_mask: int, limit: int = 0) -> List[Tuple[int, str, int]]:
        if role_mask in self.role_pools:
            return self.role_pools[role_mask]
        cursor = self.conn.cursor()
        query = "SELECT mol_id, smiles, role_mask FROM molecules WHERE (role_mask & ?) = ?"
        if limit > 0:
            query += f" LIMIT {limit}"
        cursor.execute(query, (role_mask, role_mask))
        results = cursor.fetchall()
        self.role_pools[role_mask] = results
        log.info(f"Loaded {len(results)} molecules for role_mask={role_mask}")
        return results

    def get_molecule_by_id(self, mol_id: int) -> Optional[Tuple[int, str, int]]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT mol_id, smiles, role_mask FROM molecules WHERE mol_id = ?", (mol_id,))
        row = cursor.fetchone()
        return row

    def count_molecules(self) -> int:
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM molecules")
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
            raise ValueError(f"Reaction {rxn_id} not found")
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

        log.info(
            f"DPEX_DJA v5: rxn={rxn_id}, "
            f"poolA={len(self.pool_A)}, poolB={len(self.pool_B_react)}, "
            f"poolC={len(self.pool_C)}, 3-comp={self.is_three_component}"
        )

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

        # Build Pop A (random)
        for _ in range(n_a * 2):  # generate extra, keep valid
            if len(self.pop_a) >= n_a:
                break
            ind = self._random_individual()
            if ind.valid:
                self.pop_a.append(ind)

        # Build Pop B (random, but try for smaller molecules)
        for _ in range(n_b * 2):
            if len(self.pop_b) >= n_b:
                break
            ind = self._random_individual()
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
        global psichic_wrapper

        valid_inds = [ind for ind in individuals if ind.valid and ind.smiles]
        if not valid_inds:
            return individuals

        smiles_list = [ind.smiles for ind in valid_inds]

        try:
            if psichic_wrapper is None:
                psichic_wrapper = PsichicWrapper()
                log.info("PsichicWrapper initialized")

            # Score against target
            psichic_wrapper.initialize_model(target_seq)
            target_df = psichic_wrapper.score_molecules(smiles_list)

            # Extract target scores
            target_scores = []
            if target_df is not None and 'predicted_binding_affinity' in target_df.columns:
                target_scores = target_df['predicted_binding_affinity'].tolist()
            elif target_df is not None and len(target_df.columns) > 0:
                # Try first numeric column
                for col in target_df.columns:
                    if target_df[col].dtype in ['float64', 'float32', 'int64']:
                        target_scores = target_df[col].tolist()
                        break

            if not target_scores:
                log.warning("No target scores extracted from PSICHIC")
                return individuals

            # Score against antitargets
            antitarget_scores_all = []
            for aseq in antitarget_seqs:
                try:
                    psichic_wrapper.initialize_model(aseq)
                    anti_df = psichic_wrapper.score_molecules(smiles_list)
                    if anti_df is not None and 'predicted_binding_affinity' in anti_df.columns:
                        antitarget_scores_all.append(anti_df['predicted_binding_affinity'].tolist())
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

            log.info(f"PSICHIC scored {len(valid_inds)} molecules")
            return individuals

        except Exception as e:
            log.error(f"PSICHIC scoring failed: {e}")
            traceback.print_exc()
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
        """Jaya update on Population A."""
        if not self.pop_a:
            return []

        scored = [ind for ind in self.pop_a if ind.score > float('-inf')]
        if len(scored) < 2:
            return self.pop_a

        best = max(scored, key=lambda x: x.score)
        worst = min(scored, key=lambda x: x.score)

        # Adaptive mutation rate
        mutation_boost = min(0.8, 0.15 + 0.15 * self.stall_count) if self.stall_count >= STALL_THRESHOLD else 0.0

        new_pop = []
        for ind in self.pop_a:
            if random.random() < mutation_boost:
                candidate = self._biased_individual()
            else:
                candidate = self._jaya_mutate(ind, best, worst)
            if candidate.valid:
                new_pop.append(candidate)
            else:
                new_pop.append(ind)
        return new_pop

    def evolve_pop_b(self) -> List[Individual]:
        """Tabu neighbourhood search on Population B."""
        new_pop = []
        for ind in self.pop_b:
            candidate = self._neighbourhood_search(ind)
            if candidate.valid:
                new_pop.append(candidate)
            else:
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
        """Track best-ever and stall counter."""
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

    def get_diverse_top_molecules(self, n: int) -> List[Individual]:
        """
        Multi-objective greedy selection:
        Maximize score while maintaining diversity (Tanimoto distance).
        This feeds the entropy bonus in the validator.
        """
        all_inds = self.pop_a + self.pop_b
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
    """Load challenge configuration from input.json."""
    with open(INPUT_PATH, "r") as f:
        d = json.load(f)

    # Merge config and challenge sections
    config = {}
    config.update(d.get("config", {}))
    config.update(d.get("challenge", {}))

    log.info(f"Config: targets={len(config.get('target_sequences', []))}, "
             f"antitargets={len(config.get('antitarget_sequences', []))}, "
             f"num_molecules={config.get('num_molecules', 100)}, "
             f"allowed_reaction={config.get('allowed_reaction', 'none')}, "
             f"min_ha={config.get('min_heavy_atoms', 20)}, "
             f"antitarget_weight={config.get('antitarget_weight', 0.9)}")
    return config


def find_db() -> str:
    """Locate molecules.sqlite in the sandbox."""
    candidates = [
        # nova_ph2 installed location
        "/usr/local/lib/python3.12/site-packages/nova_ph2/combinatorial_db/molecules.sqlite",
        # Other common paths
        "/workspace/combinatorial_db/molecules.sqlite",
        "/app/combinatorial_db/molecules.sqlite",
        "/combinatorial_db/molecules.sqlite",
    ]
    for c in candidates:
        if os.path.isfile(c):
            log.info(f"Database found: {c}")
            return c

    # Glob search
    for root_dir in ["/usr/local/lib", "/workspace", "/app", "/opt", "/"]:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            if "molecules.sqlite" in filenames:
                path = os.path.join(dirpath, "molecules.sqlite")
                log.info(f"Database found via walk: {path}")
                return path
            # Don't go too deep
            if dirpath.count(os.sep) > 5:
                dirnames.clear()

    raise FileNotFoundError("Cannot find molecules.sqlite")


def write_output(molecules: List[str]):
    """Write result.json."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result = {"molecules": molecules}
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log.info(f"Wrote {len(molecules)} molecules to {OUTPUT_PATH}")


def main():
    start_time = time.time()
    log.info("=" * 60)
    log.info("DPEX_DJA Blueprint Miner v5 — starting")
    log.info("=" * 60)

    # Environment diagnostic
    log.info(f"Python: {sys.version}")
    log.info(f"PSICHIC: {HAS_PSICHIC}, RDKit: {HAS_RDKIT}, CombDB: {HAS_COMB_DB}")
    log.info(f"CUDA: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    try:
        import torch
        log.info(f"PyTorch: {torch.__version__}, CUDA avail: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        log.info("PyTorch not available")

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

    while time.time() < deadline:
        iteration += 1
        elapsed = time.time() - start_time
        remaining = deadline - time.time()
        log.info(f"\n--- Iteration {iteration} | elapsed={elapsed:.0f}s | remaining={remaining:.0f}s ---")

        # Score populations
        if HAS_PSICHIC and target_seq:
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

        # Update best tracker
        optimizer.update_best()

        # Write current best to output (continuously updated)
        top = optimizer.get_diverse_top_molecules(num_molecules)
        if top:
            write_output([ind.name for ind in top])
            scored_top = [ind for ind in top if ind.score > float('-inf')]
            if scored_top:
                avg_score = sum(ind.score for ind in scored_top) / len(scored_top)
                log.info(f"Top {len(top)} avg={avg_score:.4f}, best={scored_top[0].score:.4f}")

        # Evolve populations
        log.info("Evolving Pop A (Jaya)...")
        optimizer.pop_a = optimizer.evolve_pop_a()

        log.info("Evolving Pop B (Neighbourhood)...")
        optimizer.pop_b = optimizer.evolve_pop_b()

        # Population exchange
        if iteration % EXCHANGE_INTERVAL == 0:
            log.info("Exchanging populations...")
            optimizer.exchange_populations()

        if time.time() >= deadline:
            break

    # 5. Final output
    top = optimizer.get_diverse_top_molecules(num_molecules)
    if top:
        write_output([ind.name for ind in top])
        scored_top = [ind for ind in top if ind.score > float('-inf')]
        if scored_top:
            log.info(f"\nFINAL: {len(top)} molecules, best={scored_top[0].score:.4f}")
        else:
            log.info(f"\nFINAL: {len(top)} molecules (unscored)")
    else:
        # Last resort: dump all valid molecules
        all_valid = [ind for ind in (optimizer.pop_a + optimizer.pop_b) if ind.valid]
        if all_valid:
            write_output([ind.name for ind in all_valid[:num_molecules]])
            log.info(f"\nFALLBACK: {min(len(all_valid), num_molecules)} molecules")
        else:
            write_output([])
            log.error("No valid molecules generated!")

    db.close()
    elapsed = time.time() - start_time
    log.info(f"DPEX_DJA v5 complete in {elapsed:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"Fatal error: {e}")
        traceback.print_exc()
        try:
            write_output([])
        except Exception:
            pass
