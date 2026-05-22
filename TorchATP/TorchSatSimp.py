import logging
import torch
import TorchUnify
from derivations import Derivation, flatDerivation
from TorchClauses import Clause, VirtualClause, decode_virtual_clause, Literal, parse_tptp_to_virtual_clause

logging.basicConfig(
    level=logging.INFO, # Change this to logging.INFO when benchmarking!
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("TorchSatSimp")

def compute_given_clause_resolvents_tensor(given_clause, processed_pool, pipeline):
    candidate_pairs = []
    parent_metadata = []

    # --- 1. Find inter-clause matches ---
    for p_clause in processed_pool:
        # --- THE SELF-RESOLUTION FIX ---
        if p_clause is given_clause:
            # Generate a fresh clone so variables don't collide
            g_str = decode_virtual_clause(given_clause, pipeline)
            dummy_tptp = f"cnf({given_clause.name}_clone, plain, ({g_str}))."
            target_p_clause = parse_tptp_to_virtual_clause(dummy_tptp, pipeline)
        else:
            target_p_clause = p_clause
        # -------------------------------
        # Unpack the tuples directly from the literals list
        for idx_g, (is_neg_g, root_g) in enumerate(given_clause.literals):
            for idx_p, (is_neg_p, root_p) in enumerate(target_p_clause.literals):
                
                # Check polarity instantly using the boolean flags!
                if is_neg_g != is_neg_p:
                    candidate_pairs.append((root_g, root_p))
                    parent_metadata.append({
                        "c1": given_clause, "lit_idx1": idx_g,
                        "c2": p_clause, "lit_idx2": idx_p
                    })

    if not candidate_pairs:
        return []

    logger.debug(f"Sending {len(candidate_pairs)} candidate pairs to GPU Unifier...")

    # --- 2. Batched GPU Unification ---
    subs, success_mask, unifier = pipeline.prove_batch_indices(
        candidate_pairs, standardize_apart=True 
    )
    
    resolvents = []
    successful_indices = torch.nonzero(success_mask).squeeze(-1).tolist()
    if type(successful_indices) == int: successful_indices = [successful_indices]
        
    logger.debug(f"GPU Unification complete. {len(successful_indices)} pairs successfully unified.")

    for idx in successful_indices:
        meta = parent_metadata[idx]
        
        # 1. Grab the specific 1D row for this winning unification
        winning_subs_row = unifier.subs[idx]
        
        # 2. Call the wrapper from TorchUnify!
        single_unifier = TorchUnify.SingleUnifierWrapper(winning_subs_row)
        
        old_roots_1 = [r for (sign, r) in meta["c1"].literals]
        old_roots_2 = [r for (sign, r) in meta["c2"].literals]
        
        # 3. Perform the pure-tensor surgery
        new_roots_1 = pipeline.instantiate_in_arena(
            old_roots_1, exclude_idx=meta["lit_idx1"], unifier=single_unifier
        )
        new_roots_2 = pipeline.instantiate_in_arena(
            old_roots_2, exclude_idx=meta["lit_idx2"], unifier=single_unifier
        )
        
        new_literals = []
        
        # Re-attach the boolean signs to the new roots for Clause 1
        new_root_idx = 0
        for i, (sign, old_root) in enumerate(meta["c1"].literals):
            if i != meta["lit_idx1"]:
                new_literals.append((sign, new_roots_1[new_root_idx]))
                new_root_idx += 1
                
        # Re-attach the boolean signs to the new roots for Clause 2
        new_root_idx = 0
        for i, (sign, old_root) in enumerate(meta["c2"].literals):
            if i != meta["lit_idx2"]:
                new_literals.append((sign, new_roots_2[new_root_idx]))
                new_root_idx += 1
        
        # Build the final VirtualClause with the tuple list
        new_clause = VirtualClause(new_literals)
        new_clause.deduplicate()
        new_clause.setDerivation(flatDerivation("resolution", [meta["c1"], meta["c2"]]))
        resolvents.append(new_clause)
        
        # EXTREMELY IMPORTANT: Only decode if DEBUG is actually enabled
        if logger.isEnabledFor(logging.DEBUG):
            readable_res = decode_virtual_clause(new_clause, pipeline)
            logger.debug(f"New Resolvent: {readable_res}")
        
    return resolvents

def compute_given_clause_factors_tensor(given_clause, pipeline):
    """
    Computes all valid factors for a single VirtualClause using the GPU.
    Utilizes the 2D substitution wrapper to prevent cross-batch contamination.
    """
    factors = []
    queries = []
    metadata = []
    
    # 1. Build the Factoring Pairs
    # Factoring attempts to unify two literals within the SAME clause that share the SAME sign
    lits = given_clause.literals
    for i in range(len(lits)):
        for j in range(i + 1, len(lits)):
            sign1, root1 = lits[i]
            sign2, root2 = lits[j]
            
            if sign1 == sign2:
                queries.append([root1, root2])
                metadata.append({
                    "lit_idx1": i, 
                    "lit_idx2": j  # This is the literal we will cut out during surgery
                })
                
    if not queries:
        return factors
        
    # 2. Push to the GPU Unifier
    subs, success_mask, unifier = pipeline.prove_batch_indices(queries)
    
    # Handle tensor nonzero safely for both single and multiple successes
    successful_indices = torch.nonzero(success_mask).squeeze(-1)
    if successful_indices.dim() == 0:
        successful_indices = [successful_indices.item()]
    else:
        successful_indices = successful_indices.tolist()
        
    # 3. Graph Surgery
    for idx in successful_indices:
        meta = metadata[idx]
        
        # Isolate the specific 1D row for this exact factorization
        winning_subs_row = unifier.subs[idx]
        
        # Wrap it for the surgery engine
        single_unifier = TorchUnify.SingleUnifierWrapper(winning_subs_row)
        
        # Extract raw integer roots
        old_roots = [r for (sign, r) in given_clause.literals]
        
        # Instantiate the new roots, cutting out the redundant second literal
        new_roots = pipeline.instantiate_in_arena(
            old_roots, exclude_idx=meta["lit_idx2"], unifier=single_unifier
        )
        
        new_literals = []
        new_root_idx = 0
        
        # Re-attach the boolean signs, skipping the cut literal
        for i, (sign, old_root) in enumerate(given_clause.literals):
            if i != meta["lit_idx2"]:
                new_literals.append((sign, new_roots[new_root_idx]))
                new_root_idx += 1
                
        # Finalize the new VirtualClause
        new_clause = VirtualClause(new_literals)
        new_clause.deduplicate()
        new_clause.setDerivation(flatDerivation("factor", [given_clause]))
        factors.append(new_clause)
        
    return factors


def run_given_clause_benchmark_tensor(tptp_strings, pipeline, max_loops=2000):
    
    # 1. Parse all strings into the Python-list memory arena
    unprocessed = [parse_tptp_to_virtual_clause(s, pipeline) for s in tptp_strings]
    processed = []
    
    # 2. THE TENSOR TRANSITION (Seal the arena and move to GPU)
    if isinstance(pipeline.parser.nodes, list):
        pipeline.parser.nodes = torch.tensor(
            pipeline.parser.nodes, dtype=torch.long, device=pipeline.device
        )
        pipeline.parser.children = torch.tensor(
            pipeline.parser.children, dtype=torch.long, device=pipeline.device
        )
        pipeline.parser.is_var_mask = torch.tensor(
            pipeline.parser.is_var_mask, dtype=torch.bool, device=pipeline.device
        )
        
    total_clauses_generated = 0
    loops = 0
    
    logger.info(f"Starting benchmark with {len(unprocessed)} initial clauses...")
    while unprocessed and loops < max_loops:
        loops += 1
        
        # Selection: Pure FIFO
        given_clause = unprocessed.pop(0)
        
        # Only pay the cost of string translation if we are actively debugging
        if logger.isEnabledFor(logging.DEBUG):
            readable_given = decode_virtual_clause(given_clause, pipeline)
            logger.debug(f"\n--- Loop {loops} ---")
            logger.debug(f"Selected Given Clause: {readable_given}")
            logger.debug(f"Unprocessed Queue Size: {len(unprocessed)}")
        
        # Activation
        processed.append(given_clause)
        
        # GPU Inferences (assuming you also update factors to pure tensors)
        new_factors = compute_given_clause_factors_tensor(given_clause, pipeline)
        new_resolvents = compute_given_clause_resolvents_tensor(given_clause, processed, pipeline)
        
        new_clauses = new_factors + new_resolvents
        total_clauses_generated += len(new_clauses)
        
        # Check for Proof
        for res in new_clauses:
            if res.is_empty():
                logger.info(f">>> Proof found in {loops} loops! <<<")
                logger.info(f"Total clauses generated: {total_clauses_generated}")
                
                # The proof is found, so we ALWAYS decode the final lineage
                print("\n---PROOF PATH ---")
                proof_path = res.orderedDerivation()
                for step in proof_path:
                    # You might need to attach the decoded string to the __repr__ of the 
                    # VirtualClause so derivations.py can print it correctly.
                    print(decode_virtual_clause(step, pipeline))
                    
                return {"status": "Theorem", "loops": loops, "clauses": total_clauses_generated}
                
        # Retention
        unprocessed.extend(new_clauses)

    logger.warning("Search space exhausted or max loops reached.")
    return {"status": "Timeout/Exhausted", "loops": loops, "clauses": total_clauses_generated}