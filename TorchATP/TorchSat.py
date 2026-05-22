import torch
import TorchUnify
from derivations import flatDerivation
# Make sure you are importing VirtualClause instead of Clause
from TorchClauses import VirtualClause, parse_tptp_to_virtual_clause, decode_virtual_clause

def compute_batched_resolvents_tensor(clause_pool, pipeline):
    queries = []
    parent_metadata = []

    pool_size = len(clause_pool)

    # O(N^2) All-to-All Generation
    for i in range(pool_size):
        for j in range(i, pool_size):
            c1 = clause_pool[i]

            # --- THE SELF-RESOLUTION FIX ---
            if i == j:
                # 1. Decode the clause back to a readable string
                c1_str = decode_virtual_clause(c1, pipeline)
                # 2. Wrap it in standard TPTP syntax so the parser accepts it
                dummy_tptp = f"cnf({c1.name}_clone, plain, ({c1_str}))."
                # 3. Parse it back into the arena! It now has completely fresh variable IDs.
                c2 = parse_tptp_to_virtual_clause(dummy_tptp, pipeline)
            else:
                c2 = clause_pool[j]
            # -------------------------------

            for idx1, (sign1, root1) in enumerate(c1.literals):
                for idx2, (sign2, root2) in enumerate(c2.literals):
                    # Prevent resolving a literal against itself in the same clause
                    if i == j and idx1 == idx2:
                        continue
                    
                    # Resolution requires OPPOSITE polarities
                    if sign1 != sign2:
                        queries.append([root1, root2])
                        parent_metadata.append({
                            "c1": c1, "lit_idx1": idx1,
                            "c2": c2, "lit_idx2": idx2
                        })
                        
    if not queries:
        return []
    
    # 1. Batched GPU Unification directly on structural indices
    subs, success_mask, unifier = pipeline.prove_batch_indices(queries)

    resolvents = []
    
    # Safely handle 0D, 1D, or empty tensors
    successful_indices = torch.nonzero(success_mask).squeeze(-1)
    if successful_indices.dim() == 0:
        successful_indices = [successful_indices.item()]
    else:
        successful_indices = successful_indices.tolist()

    # 2. Graph Surgery via the Memory Arena
    for idx in successful_indices:
        meta = parent_metadata[idx]

        # Isolate the exact 1D row for this pair to avoid batch contamination
        winning_subs_row = unifier.subs[idx]
        single_unifier = TorchUnify.SingleUnifierWrapper(winning_subs_row)
        
        old_roots_1 = [r for (sign, r) in meta["c1"].literals]
        old_roots_2 = [r for (sign, r) in meta["c2"].literals]
        
        # Instantiate in arena (automatically applying the wrapper substitutions)
        new_roots_1 = pipeline.instantiate_in_arena(
            old_roots_1, exclude_idx=meta["lit_idx1"], unifier=single_unifier
        )
        new_roots_2 = pipeline.instantiate_in_arena(
            old_roots_2, exclude_idx=meta["lit_idx2"], unifier=single_unifier
        )

        new_literals = []
        
        # Re-attach boolean signs for Clause 1
        new_root_idx = 0
        for i, (sign, _) in enumerate(meta["c1"].literals):
            if i != meta["lit_idx1"]:
                new_literals.append((sign, new_roots_1[new_root_idx]))
                new_root_idx += 1
                
        # Re-attach boolean signs for Clause 2
        new_root_idx = 0
        for i, (sign, _) in enumerate(meta["c2"].literals):
            if i != meta["lit_idx2"]:
                new_literals.append((sign, new_roots_2[new_root_idx]))
                new_root_idx += 1

        new_clause = VirtualClause(new_literals)
        new_clause.deduplicate()
        
        new_clause.setDerivation(
            flatDerivation("resolution", [meta["c1"], meta["c2"]])
        )

        resolvents.append(new_clause)

    return resolvents

def compute_batched_factors_tensor(clause_pool, pipeline):
    queries = []
    parent_metadata = []

    # --- 1. Build the Intra-Clause Pairs ---
    for clause in clause_pool:
        num_lits = len(clause.literals)
        
        for i in range(num_lits):
            for j in range(i + 1, num_lits):
                sign1, root1 = clause.literals[i]
                sign2, root2 = clause.literals[j]

                # Factoring requires the SAME polarity
                if sign1 == sign2:
                    queries.append([root1, root2])
                    parent_metadata.append({
                        "clause": clause,
                        "lit_idx1": i,
                        "lit_idx2": j  # literal to drop
                    })

    if not queries:
        return []

    # --- 2. Batched GPU Unification ---
    subs, success_mask, unifier = pipeline.prove_batch_indices(queries)

    factors = []
    successful_indices = torch.nonzero(success_mask).squeeze(-1)
    if successful_indices.dim() == 0:
        successful_indices = [successful_indices.item()]
    else:
        successful_indices = successful_indices.tolist()

    # --- 3. Graph Surgery ---
    for idx in successful_indices:
        meta = parent_metadata[idx]
        clause = meta["clause"]

        winning_subs_row = unifier.subs[idx]
        single_unifier = TorchUnify.SingleUnifierWrapper(winning_subs_row)

        old_roots = [r for (sign, r) in clause.literals]

        # Instantiate, excluding the redundant literal
        new_roots = pipeline.instantiate_in_arena(
            old_roots, exclude_idx=meta["lit_idx2"], unifier=single_unifier
        )

        new_literals = []
        new_root_idx = 0
        
        for i, (sign, _) in enumerate(clause.literals):
            if i != meta["lit_idx2"]:
                new_literals.append((sign, new_roots[new_root_idx]))
                new_root_idx += 1

        new_clause = VirtualClause(new_literals)
        new_clause.deduplicate()
        
        new_clause.setDerivation(
            flatDerivation("factor", [clause])
        )

        factors.append(new_clause)

    return factors

def run_neural_prover_tensor(tptp_strings, pipeline, neural_network, top_k=1000):
    # INITIALIZATION FIX: Must parse directly into the VirtualClause memory arena!
    current_pool = [parse_tptp_to_virtual_clause(s, pipeline) for s in tptp_strings]
    generation = 0
    
    print(f"Starting proof search with {len(current_pool)} initial clauses...")
    
    while True:
        generation += 1
        print(f"--- Generation {generation} ---")
        
        # 1. Batched GPU Resolution (All-to-All)
        new_resolvents = compute_batched_resolvents_tensor(current_pool, pipeline)
        
        # 2. Batched GPU Factoring
        new_factors = compute_batched_factors_tensor(current_pool, pipeline)
        
        # Combine all generated clauses
        all_generated = new_resolvents + new_factors
        
        if not all_generated:
            print("Search space exhausted. Proof failed.")
            return False
            
        # Check for the Empty Clause
        for res in all_generated:
            if res.is_empty():
                print(">>> EMPTY CLAUSE DERIVED! Proof found! <<<")
                
                # You may need a custom string formatter here if you want 
                # the trace printed logically instead of as integer tuples.
                proof_path = res.orderedDerivation()
                for step in proof_path:
                    print(step)
                return True
                
        # NEURAL NETWORK SCORING FIX
        # If your NN takes string embeddings (Transformers), you must decode them first.
        # If your NN takes structural integers (Graph Neural Networks), send 'all_generated' directly.
        # Assuming a text-based transformer for now:
        clause_strings = [pipeline.decode_virtual_clause(c) for c in all_generated]
        scores = neural_network.score_clauses(clause_strings)
        
        # Sort and slice
        scored_pairs = list(zip(scores, all_generated))
        scored_pairs.sort(key=lambda x: x[0], reverse=True)
        
        best_new_clauses = [clause for score, clause in scored_pairs[:top_k]]
        
        current_pool.extend(best_new_clauses)