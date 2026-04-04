#!/usr/bin/env python3
"""
miner.py — DPEX_DJA Blueprint Miner for SN68 Nova
Dual-Population Discrete Jaya Algorithm with ComponentRanker

Strategy:
  1. Pre-fetch all reactant pools from SAVI combinatorial DB
  2. Initialize dual populations (exploration + exploitation)
  3. Score with PSICHIC via validator scoring module
  4. Use Jaya update rule to evolve populations toward better reactant combos
  5. ComponentRanker tracks EMA quality of individual reactants
  6. Continuously write best molecules to /output/result.json
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
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict

# ---------------------------------------------------------------------------
# Path discovery — the Docker sandbox has nova-blueprint installed; we need
# to locate the combinatorial DB and the scoring / PSICHIC modules.
# ---------------------------------------------------------------------------
_SEARCH_ROOTS = [
    Path("/workspace"),
    Path("/"),
    Path("/app"),
    Path("/nova-blueprint"),
    Path("/opt"),
    Path(os.path.dirname(os.path.abspath(__file__))),
]

def _find_path(name: str, is_dir: bool = True) -> Optional[str]:
    """Find a file or directory by name across known root paths."""
    for root in _SEARCH_ROOTS:
        for depth in range(4):
            for p in root.glob("/".join(["*"] * depth + [name])):
                if is_dir and p.is_dir():
                    return str(p)
                if not is_dir and p.is_file():
                    return str(p)
    return None


def _find_db() -> str:
    """Locate molecules.sqlite in the sandbox."""
    # Direct known paths first
    candidates = [
        "/combinatorial_db/molecules.sqlite",
        "/workspace/combinatorial_db/molecules.sqlite",
        "/app/combinatorial_db/molecules.sqlite",
        "/nova-blueprint/combinatorial_db/molecules.sqlite",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    # Glob search
    found = _find_path("molecules.sqlite", is_dir=False)
    if found:
        return found
    raise FileNotFoundError("Cannot find molecules.sqlite in Docker sandbox")


def _setup_imports():
    """Add necessary paths to sys.path so we can import validator scoring + PSICHIC."""
    dirs_to_add = set()
    # Try to find the nova-blueprint root (parent of combinatorial_db, neurons, PSICHIC)
    for marker in ["neurons", "PSICHIC", "combinatorial_db"]:
        d = _find_path(marker, is_dir=True)
        if d:
            parent = str(Path(d).parent)
            dirs_to_add.add(parent)
    # Also add common locations
    for p in ["/workspace", "/app", "/nova-blueprint", "/"]:
        if os.path.isdir(p):
            dirs_to_add.add(p)
    for d in dirs_to_add:
        if d not in sys.path:
            sys.path.insert(0, d)


_setup_imports()

# Now import what we need
try:
    import bittensor as bt
except ImportError:
    # Minimal fallback logging if bt not available
    class _FallbackLog:
        @staticmethod
        def info(msg): print(f"[INFO] {msg}")
        @staticmethod
        def warning(msg): print(f"[WARN] {msg}")
        @staticmethod
        def error(msg): print(f"[ERROR] {msg}")
    class bt:
        logging = _FallbackLog()

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

# Import PSICHIC scoring from the validator module
try:
    from neurons.validator.scoring import score_molecules_json
    import neurons.validator.scoring as scoring_module
    HAS_PSICHIC_SCORING = True
except ImportError:
    try:
        from validator.scoring import score_molecules_json
        import validator.scoring as scoring_module
        HAS_PSICHIC_SCORING = True
    except ImportError:
        HAS_PSICHIC_SCORING = False

# Import reaction utilities
try:
    from combinatorial_db.reactions import (
        get_reaction_info,
        get_smiles_from_reaction,
        validate_and_order_reactants,
        perform_smarts_reaction,
        combine_triazole_synthons,
    )
    HAS_REACTIONS = True
except ImportError:
    HAS_REACTIONS = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INPUT_PATH = "/workspace/input.json"
OUTPUT_PATH = "/output/result.json"
SCRATCH_DIR = "/tmp/dpex_dja"

POP_A_SIZE = 300     # Global exploration population
POP_B_SIZE = 100     # Local exploitation population
BATCH_SIZE = 500     # Molecules to sample per iteration
STALL_THRESHOLD = 5  # Iterations without improvement before mutation boost
EMA_ALPHA = 0.3      # Exponential moving average smoothing for ComponentRanker
EXCHANGE_INTERVAL = 3 # Population exchange every N iterations
TIME_RESERVE_SEC = 30 # Stop this many seconds before deadline


# ===========================================================================
# ComponentRanker — tracks EMA quality of individual reactants
# ===========================================================================
class ComponentRanker:
    """
    Maintains an exponential moving average (EMA) of scores for each reactant
    (identified by mol_id). This lets us bias sampling toward reactants that
    have historically contributed to high-scoring products.
    """
    def __init__(self, alpha: float = EMA_ALPHA):
        self.alpha = alpha
        self.ema: Dict[int, float] = {}       # mol_id → EMA score
        self.count: Dict[int, int] = {}        # mol_id → times seen

    def update(self, mol_ids: List[int], score: float):
        """Update EMA for each reactant that contributed to a product."""
        for mid in mol_ids:
            if mid not in self.ema:
                self.ema[mid] = score
                self.count[mid] = 1
            else:
                self.ema[mid] = self.alpha * score + (1 - self.alpha) * self.ema[mid]
                self.count[mid] += 1

    def get_weights(self, mol_ids: List[int]) -> List[float]:
        """
        Return sampling weights for a list of mol_ids.
        Higher EMA → higher weight. Unseen mol_ids get neutral weight.
        """
        scores = []
        for mid in mol_ids:
            if mid in self.ema:
                scores.append(self.ema[mid])
            else:
                scores.append(0.0)  # neutral
        # Shift so minimum is 0, then add small epsilon
        if not scores:
            return []
        min_s = min(scores)
        weights = [s - min_s + 1e-6 for s in scores]
        return weights

    def top_k(self, k: int = 50) -> List[int]:
        """Return top-k mol_ids by EMA score."""
        sorted_ids = sorted(self.ema.keys(), key=lambda x: self.ema[x], reverse=True)
        return sorted_ids[:k]


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
        """Load all available reactions."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT rxn_id, smarts, roleA, roleB, roleC FROM reactions")
        return cursor.fetchall()

    def get_reaction_by_id(self, rxn_id: int) -> Optional[Tuple]:
        """Get reaction info by ID."""
        for r in self.reactions:
            if r[0] == rxn_id:
                return r
        return None

    def get_molecules_by_role(self, role_mask: int) -> List[Tuple[int, str, int]]:
        """Get all molecules matching a role mask, with caching."""
        if role_mask in self.role_pools:
            return self.role_pools[role_mask]
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT mol_id, smiles, role_mask FROM molecules WHERE (role_mask & ?) = ?",
            (role_mask, role_mask)
        )
        results = cursor.fetchall()
        self.role_pools[role_mask] = results
        bt.logging.info(f"Loaded {len(results)} molecules for role_mask={role_mask}")
        return results

    def close(self):
        self.conn.close()


# ===========================================================================
# Molecule Generation & Validation
# ===========================================================================
def compute_product_smiles_safe(
    rxn_id: int, smarts: str,
    roleA: int, roleB: int, roleC: Optional[int],
    molA: Tuple, molB: Tuple, molC: Optional[Tuple] = None
) -> Optional[str]:
    """Compute product SMILES from reactants, handling all reaction types."""
    if not HAS_REACTIONS:
        return None
    try:
        _, smiA, rmA = molA
        _, smiB, rmB = molB
        if roleC and roleC != 0 and molC:
            _, smiC, rmC = molC
            v = validate_and_order_reactants(smiA, smiB, rmA, rmB, roleA, roleB, smiC, rmC, roleC)
            if not all(v):
                return None
            r1, r2, r3 = v
            if rxn_id == 3:
                triazole_cooh = combine_triazole_synthons(r1, r2)
                if not triazole_cooh:
                    return None
                amide_smarts = "[C:1](=O)[OH].[N:2]>>[C:1](=O)[N:2]"
                return perform_smarts_reaction(triazole_cooh, r3, amide_smarts)
            if rxn_id == 5:
                suzuki_br = "[#6:1][Br].[#6:2][B]([OH])[OH]>>[#6:1][#6:2]"
                suzuki_cl = "[#6:1][Cl].[#6:2][B]([OH])[OH]>>[#6:1][#6:2]"
                intermediate = perform_smarts_reaction(r1, r2, suzuki_br)
                if not intermediate:
                    return None
                return perform_smarts_reaction(intermediate, r3, suzuki_cl)
            return None
        else:
            r1, r2 = validate_and_order_reactants(smiA, smiB, rmA, rmB, roleA, roleB)
            if not r1 or not r2:
                return None
            if rxn_id == 1:
                return combine_triazole_synthons(r1, r2)
            return perform_smarts_reaction(r1, r2, smarts)
    except Exception:
        return None


def validate_molecule(smiles: str, config: dict) -> bool:
    """Check if a molecule meets the config constraints."""
    if not smiles or not HAS_RDKIT:
        return False
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
        return True
    except Exception:
        return False


# ===========================================================================
# Individual — represents a single molecule solution
# ===========================================================================
class Individual:
    """A molecule defined by its reactant combination."""
    __slots__ = ['rxn_id', 'mol_ids', 'name', 'smiles', 'score', 'valid']

    def __init__(self, rxn_id: int, mol_ids: List[int], name: str = "",
                 smiles: str = "", score: float = float('-inf'), valid: bool = False):
        self.rxn_id = rxn_id
        self.mol_ids = mol_ids
        self.name = name
        self.smiles = smiles
        self.score = score
        self.valid = valid

    def __repr__(self):
        return f"Individual({self.name}, score={self.score:.4f})"


# ===========================================================================
# DPEX_DJA Optimizer
# ===========================================================================
class DPEX_DJA:
    """
    Dual-Population Discrete Jaya Algorithm.

    Pop A (large): global exploration via DJA update rule
    Pop B (small): local refinement via tabu-enhanced neighbourhood search
    """

    def __init__(self, db: SAVIDatabase, config: dict, rxn_id: int):
        self.db = db
        self.config = config
        self.rxn_id = rxn_id
        self.ranker = ComponentRanker()

        # Get reaction info
        rxn_info = db.get_reaction_by_id(rxn_id)
        if not rxn_info:
            raise ValueError(f"Reaction {rxn_id} not found in database")
        self.smarts = rxn_info[1]
        self.roleA = rxn_info[2]
        self.roleB = rxn_info[3]
        self.roleC = rxn_info[4] if rxn_info[4] and rxn_info[4] != 0 else None
        self.is_three_component = self.roleC is not None

        # Pre-fetch reactant pools
        self.pool_A = db.get_molecules_by_role(self.roleA)
        self.pool_B_react = db.get_molecules_by_role(self.roleB)
        self.pool_C = db.get_molecules_by_role(self.roleC) if self.is_three_component else []

        # Index pools by mol_id for fast lookup
        self.idx_A = {m[0]: m for m in self.pool_A}
        self.idx_B = {m[0]: m for m in self.pool_B_react}
        self.idx_C = {m[0]: m for m in self.pool_C} if self.is_three_component else {}

        # Populations
        self.pop_a: List[Individual] = []
        self.pop_b: List[Individual] = []

        # Best ever
        self.best_ever: Optional[Individual] = None

        # Stall counter for anti-plateau
        self.stall_count = 0
        self.last_best_score = float('-inf')

        # Tabu set for Pop B
        self.tabu_set: set = set()

        bt.logging.info(
            f"DPEX_DJA initialized: rxn={rxn_id}, "
            f"poolA={len(self.pool_A)}, poolB={len(self.pool_B_react)}, "
            f"poolC={len(self.pool_C)}, 3-comp={self.is_three_component}"
        )

    def _make_individual(self, molA: Tuple, molB: Tuple, molC: Optional[Tuple] = None) -> Individual:
        """Create an Individual from reactant tuples."""
        idA, _, _ = molA
        idB, _, _ = molB
        if self.is_three_component and molC:
            idC, _, _ = molC
            name = f"rxn:{self.rxn_id}:{idA}:{idB}:{idC}"
            mol_ids = [idA, idB, idC]
        else:
            name = f"rxn:{self.rxn_id}:{idA}:{idB}"
            mol_ids = [idA, idB]

        smiles = compute_product_smiles_safe(
            self.rxn_id, self.smarts,
            self.roleA, self.roleB, self.roleC,
            molA, molB, molC
        )
        valid = smiles is not None and validate_molecule(smiles, self.config)
        return Individual(self.rxn_id, mol_ids, name, smiles or "", float('-inf'), valid)

    def _random_individual(self) -> Individual:
        """Generate a random valid individual."""
        for _ in range(20):  # Try up to 20 times
            molA = random.choice(self.pool_A)
            molB = random.choice(self.pool_B_react)
            molC = random.choice(self.pool_C) if self.is_three_component else None
            ind = self._make_individual(molA, molB, molC)
            if ind.valid:
                return ind
        # Return invalid individual as last resort
        molA = random.choice(self.pool_A)
        molB = random.choice(self.pool_B_react)
        molC = random.choice(self.pool_C) if self.is_three_component else None
        return self._make_individual(molA, molB, molC)

    def _biased_individual(self) -> Individual:
        """Generate individual biased toward high-quality reactants via ComponentRanker."""
        for _ in range(20):
            # Biased selection for pool A
            pool_a_ids = [m[0] for m in self.pool_A]
            weights_a = self.ranker.get_weights(pool_a_ids)
            if weights_a and sum(weights_a) > 0:
                molA = random.choices(self.pool_A, weights=weights_a, k=1)[0]
            else:
                molA = random.choice(self.pool_A)

            # Biased selection for pool B
            pool_b_ids = [m[0] for m in self.pool_B_react]
            weights_b = self.ranker.get_weights(pool_b_ids)
            if weights_b and sum(weights_b) > 0:
                molB = random.choices(self.pool_B_react, weights=weights_b, k=1)[0]
            else:
                molB = random.choice(self.pool_B_react)

            molC = None
            if self.is_three_component:
                pool_c_ids = [m[0] for m in self.pool_C]
                weights_c = self.ranker.get_weights(pool_c_ids)
                if weights_c and sum(weights_c) > 0:
                    molC = random.choices(self.pool_C, weights=weights_c, k=1)[0]
                else:
                    molC = random.choice(self.pool_C)

            ind = self._make_individual(molA, molB, molC)
            if ind.valid:
                return ind
        return self._random_individual()

    def initialize_populations(self, n_a: int = POP_A_SIZE, n_b: int = POP_B_SIZE):
        """Create initial random populations."""
        bt.logging.info(f"Initializing Pop A ({n_a}) and Pop B ({n_b})...")
        self.pop_a = [self._random_individual() for _ in range(n_a)]
        self.pop_b = [self._random_individual() for _ in range(n_b)]
        # Filter to valid only, refill
        self.pop_a = [ind for ind in self.pop_a if ind.valid]
        self.pop_b = [ind for ind in self.pop_b if ind.valid]
        while len(self.pop_a) < n_a:
            ind = self._random_individual()
            if ind.valid:
                self.pop_a.append(ind)
        while len(self.pop_b) < n_b:
            ind = self._random_individual()
            if ind.valid:
                self.pop_b.append(ind)
        bt.logging.info(f"Populations ready: A={len(self.pop_a)}, B={len(self.pop_b)}")

    def _jaya_mutate(self, current: Individual, best: Individual, worst: Individual) -> Individual:
        """
        Discrete Jaya update: for each reactant slot, probabilistically
        move toward best's reactant and away from worst's reactant.
        """
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
            idx = idxs[slot_idx]

            r1 = random.random()
            r2 = random.random()

            if r1 > 0.5 and best_id in idx:
                # Move toward best: use best's reactant
                new_mol_ids.append(best_id)
            elif r2 > 0.5 and worst_id != curr_id:
                # Move away from worst: pick a biased random (not worst)
                candidates = [m for m in pool if m[0] != worst_id]
                if candidates:
                    pool_ids = [m[0] for m in candidates]
                    weights = self.ranker.get_weights(pool_ids)
                    if weights and sum(weights) > 0:
                        choice = random.choices(candidates, weights=weights, k=1)[0]
                    else:
                        choice = random.choice(candidates)
                    new_mol_ids.append(choice[0])
                else:
                    new_mol_ids.append(curr_id)
            else:
                # Keep current
                new_mol_ids.append(curr_id)

        # Build new individual from selected mol_ids
        molA = self.idx_A.get(new_mol_ids[0])
        molB = self.idx_B.get(new_mol_ids[1])
        molC = self.idx_C.get(new_mol_ids[2]) if self.is_three_component and len(new_mol_ids) > 2 else None

        if molA and molB:
            return self._make_individual(molA, molB, molC)
        return self._random_individual()

    def _neighbourhood_search(self, ind: Individual) -> Individual:
        """
        Local search for Pop B: swap one reactant slot with a nearby reactant.
        Uses tabu to avoid revisiting.
        """
        pools = [self.pool_A, self.pool_B_react]
        idxs = [self.idx_A, self.idx_B]
        if self.is_three_component:
            pools.append(self.pool_C)
            idxs.append(self.idx_C)

        # Pick a random slot to mutate
        slot = random.randint(0, len(ind.mol_ids) - 1)
        pool = pools[slot]

        # Biased selection toward high-quality reactants
        pool_ids = [m[0] for m in pool]
        weights = self.ranker.get_weights(pool_ids)
        if weights and sum(weights) > 0:
            new_reactant = random.choices(pool, weights=weights, k=1)[0]
        else:
            new_reactant = random.choice(pool)

        new_mol_ids = list(ind.mol_ids)
        new_mol_ids[slot] = new_reactant[0]

        # Check tabu
        key = tuple(new_mol_ids)
        if key in self.tabu_set:
            return self._biased_individual()
        self.tabu_set.add(key)

        molA = self.idx_A.get(new_mol_ids[0])
        molB = self.idx_B.get(new_mol_ids[1])
        molC = self.idx_C.get(new_mol_ids[2]) if self.is_three_component and len(new_mol_ids) > 2 else None
        if molA and molB:
            return self._make_individual(molA, molB, molC)
        return self._biased_individual()

    def score_population(self, population: List[Individual], config: dict) -> List[Individual]:
        """Score a list of Individuals using PSICHIC."""
        if not HAS_PSICHIC_SCORING:
            bt.logging.warning("PSICHIC scoring not available, using random scores")
            for ind in population:
                ind.score = random.uniform(-2, 2) if ind.valid else float('-inf')
            return population

        # Filter to valid individuals
        valid_inds = [ind for ind in population if ind.valid and ind.smiles]
        if not valid_inds:
            return population

        # Build sampler-format JSON for scoring
        molecules = [ind.name for ind in valid_inds]
        sampler_data = {"molecules": molecules}

        os.makedirs(SCRATCH_DIR, exist_ok=True)
        sampler_path = os.path.join(SCRATCH_DIR, "batch.json")
        with open(sampler_path, "w") as f:
            json.dump(sampler_data, f)

        try:
            target_seqs = config.get("target_sequences", [])
            antitarget_seqs = config.get("antitarget_sequences", [])

            score_dict = score_molecules_json(
                sampler_path, target_seqs, antitarget_seqs, config
            )

            if score_dict:
                # Extract per-molecule scores
                for uid, data in score_dict.items():
                    targets = data.get("ps_target_scores", [])
                    antitargets = data.get("ps_antitarget_scores", [])
                    antitarget_weight = config.get("antitarget_weight", 0.9)

                    for mol_idx, ind in enumerate(valid_inds):
                        if mol_idx >= len(targets[0]) if targets and targets[0] else True:
                            continue
                        # Average target scores
                        t_scores = [t[mol_idx] for t in targets if mol_idx < len(t)]
                        avg_target = sum(t_scores) / len(t_scores) if t_scores else 0

                        # Average antitarget scores
                        a_scores = [a[mol_idx] for a in antitargets if mol_idx < len(a)]
                        avg_antitarget = sum(a_scores) / len(a_scores) if a_scores else 0

                        ind.score = avg_target - antitarget_weight * avg_antitarget

                        # Update ComponentRanker
                        self.ranker.update(ind.mol_ids, ind.score)
                    break  # Only one UID in our scoring
        except Exception as e:
            bt.logging.error(f"Scoring error: {e}")
            traceback.print_exc()

        return population

    def evolve_pop_a(self) -> List[Individual]:
        """Jaya update on Population A (global exploration)."""
        if not self.pop_a:
            return []

        scored = [ind for ind in self.pop_a if ind.score > float('-inf')]
        if len(scored) < 2:
            return self.pop_a

        best = max(scored, key=lambda x: x.score)
        worst = min(scored, key=lambda x: x.score)

        # Mutation probability increases when stalled
        mutation_boost = min(0.8, 0.1 + 0.1 * self.stall_count) if self.stall_count >= STALL_THRESHOLD else 0.0

        new_pop = []
        for ind in self.pop_a:
            if random.random() < mutation_boost:
                # Anti-stall: inject random biased individual
                candidate = self._biased_individual()
            else:
                candidate = self._jaya_mutate(ind, best, worst)

            if candidate.valid:
                new_pop.append(candidate)
            else:
                new_pop.append(ind)  # Keep current if mutation invalid

        return new_pop

    def evolve_pop_b(self) -> List[Individual]:
        """Tabu neighbourhood search on Population B (local exploitation)."""
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

        # Best of A → B, best of B → A
        scored_a = [ind for ind in self.pop_a if ind.score > float('-inf')]
        scored_b = [ind for ind in self.pop_b if ind.score > float('-inf')]

        if scored_a:
            best_a = max(scored_a, key=lambda x: x.score)
            # Replace worst in B
            if self.pop_b:
                worst_b_idx = min(range(len(self.pop_b)), key=lambda i: self.pop_b[i].score)
                if best_a.score > self.pop_b[worst_b_idx].score:
                    self.pop_b[worst_b_idx] = best_a

        if scored_b:
            best_b = max(scored_b, key=lambda x: x.score)
            # Replace worst in A
            if self.pop_a:
                worst_a_idx = min(range(len(self.pop_a)), key=lambda i: self.pop_a[i].score)
                if best_b.score > self.pop_a[worst_a_idx].score:
                    self.pop_a[worst_a_idx] = best_b

    def get_top_molecules(self, n: int) -> List[Individual]:
        """Return top n unique molecules across both populations."""
        all_inds = self.pop_a + self.pop_b
        scored = [ind for ind in all_inds if ind.valid and ind.score > float('-inf')]

        # Deduplicate by name
        seen = set()
        unique = []
        for ind in sorted(scored, key=lambda x: x.score, reverse=True):
            if ind.name not in seen:
                seen.add(ind.name)
                unique.append(ind)

        return unique[:n]

    def update_best(self):
        """Track the best-ever individual and stall counter."""
        all_scored = [ind for ind in (self.pop_a + self.pop_b) if ind.score > float('-inf')]
        if not all_scored:
            return

        current_best = max(all_scored, key=lambda x: x.score)
        if self.best_ever is None or current_best.score > self.best_ever.score:
            self.best_ever = current_best
            self.stall_count = 0
            bt.logging.info(f"New best: {current_best.name} score={current_best.score:.4f}")
        else:
            self.stall_count += 1


# ===========================================================================
# Main
# ===========================================================================
def load_config() -> dict:
    """Load challenge configuration from input.json."""
    with open(INPUT_PATH, "r") as f:
        d = json.load(f)
    config = {**d.get("config", {}), **d.get("challenge", {})}
    bt.logging.info(f"Config loaded: targets={len(config.get('target_sequences', []))}, "
                    f"antitargets={len(config.get('antitarget_sequences', []))}, "
                    f"num_molecules={config.get('num_molecules', 100)}, "
                    f"allowed_reaction={config.get('allowed_reaction', 'unknown')}")
    return config


def write_output(molecules: List[str]):
    """Write result.json in the required format."""
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    result = {"molecules": molecules}
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    bt.logging.info(f"Wrote {len(molecules)} molecules to {OUTPUT_PATH}")


def main():
    start_time = time.time()
    bt.logging.info("=" * 60)
    bt.logging.info("DPEX_DJA Blueprint Miner v2 — starting")
    bt.logging.info("=" * 60)

    # 1. Load config
    config = load_config()
    time_budget = config.get("time_budget_sec", 900)
    num_molecules = config.get("num_molecules", 100)
    allowed_reaction = config.get("allowed_reaction", "rxn:0")
    rxn_id = int(allowed_reaction.split(":")[-1])

    deadline = start_time + time_budget - TIME_RESERVE_SEC

    # 2. Find and connect to SAVI database
    db_path = _find_db()
    bt.logging.info(f"Database: {db_path}")
    db = SAVIDatabase(db_path)

    # 3. Initialize optimizer
    optimizer = DPEX_DJA(db, config, rxn_id)
    optimizer.initialize_populations()

    # 4. Main optimization loop
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        elapsed = time.time() - start_time
        remaining = deadline - time.time()
        bt.logging.info(f"\n--- Iteration {iteration} | elapsed={elapsed:.0f}s | remaining={remaining:.0f}s ---")

        # Score both populations
        bt.logging.info("Scoring Pop A...")
        optimizer.pop_a = optimizer.score_population(optimizer.pop_a, config)
        if time.time() >= deadline:
            break

        bt.logging.info("Scoring Pop B...")
        optimizer.pop_b = optimizer.score_population(optimizer.pop_b, config)
        if time.time() >= deadline:
            break

        # Update best tracker
        optimizer.update_best()

        # Write current best to output (continuously updated)
        top = optimizer.get_top_molecules(num_molecules)
        if top:
            write_output([ind.name for ind in top])
            avg_score = sum(ind.score for ind in top) / len(top)
            bt.logging.info(f"Top {len(top)} avg={avg_score:.4f}, best={top[0].score:.4f}")

        # Evolve populations
        bt.logging.info("Evolving Pop A (Jaya)...")
        optimizer.pop_a = optimizer.evolve_pop_a()

        bt.logging.info("Evolving Pop B (Neighbourhood)...")
        optimizer.pop_b = optimizer.evolve_pop_b()

        # Population exchange
        if iteration % EXCHANGE_INTERVAL == 0:
            bt.logging.info("Exchanging populations...")
            optimizer.exchange_populations()

        # Check time
        if time.time() >= deadline:
            break

    # 5. Final output
    top = optimizer.get_top_molecules(num_molecules)
    if top:
        write_output([ind.name for ind in top])
        bt.logging.info(f"\nFinal: {len(top)} molecules, best={top[0].score:.4f}")
    else:
        # Fallback: write whatever we have
        all_valid = [ind for ind in (optimizer.pop_a + optimizer.pop_b) if ind.valid]
        if all_valid:
            write_output([ind.name for ind in all_valid[:num_molecules]])
        else:
            write_output([])

    db.close()
    elapsed = time.time() - start_time
    bt.logging.info(f"DPEX_DJA complete in {elapsed:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        bt.logging.error(f"Fatal error: {e}")
        traceback.print_exc()
        # Write empty output on fatal error
        try:
            write_output([])
        except Exception:
            pass
