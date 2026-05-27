import logging
import torch
import TorchUnifyPlus
from derivations import Derivation, flatDerivation
from TorchClausesPlus import Clause, VirtualClause, decode_virtual_clause, Literal, parse_tptp_to_virtual_clause

logging.basicConfig(
    level=logging.INFO, # Change this to logging.INFO when benchmarking!
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("TorchSatSimpBatchPlus")

def compute_given_clause_resolvents_tensor(given_clause, processed_pool, pipeline):
    candidate_pairs = []
    parent_metadata = []

    # --- 1. Find inter-clause matches ---
    for p_clause in processed_pool:
        # --- THE SELF-RESOLUTION FIX ---
        if p_clause is given_clause:
            g_str = decode_virtual_clause(given_clause, pipeline)
            dummy_tptp = f"cnf({given_clause.name}_clone, plain, ({g_str}))."
            target_p_clause = parse_tptp_to_virtual_clause(dummy_tptp, pipeline)
        else:
            target_p_clause = p_clause
        
        logger.debug(f"given clause literals : {given_clause.literals}")
        logger.debug(f"target clause literals : {target_p_clause.literals}")
        for idx_g, (is_neg_g, root_g) in enumerate(given_clause.literals):
            for idx_p, (is_neg_p, root_p) in enumerate(target_p_clause.literals):
                if is_neg_g != is_neg_p:
                    candidate_pairs.append((root_g, root_p))
                    parent_metadata.append({
                        "c1": given_clause, "lit_idx1": idx_g,
                        "c2": target_p_clause, "lit_idx2": idx_p
                    })
    logger.debug(f"Meta-data : {parent_metadata} ")
    if not candidate_pairs:
        return []
    logger.debug(f"Candidate Pairs: {candidate_pairs}")
    logger.debug(f"Sending {len(candidate_pairs)} candidate pairs to GPU Unifier...")

    # --- 2. Batched GPU Unification ---
    subs, success_mask, unifier = pipeline.prove_batch_indices(
        candidate_pairs, standardize_apart=True 
    )

    logger.debug(f"These are the nodes: {unifier.nodes}")
    
    successful_indices = torch.nonzero(success_mask).squeeze(-1)
    if successful_indices.dim() == 0:
        successful_indices = [successful_indices.item()]
    else:
        successful_indices = successful_indices.tolist()

    logger.debug(f"GPU Unification complete. {len(successful_indices)} pairs successfully unified.")

    # --- 3. THE PACKING PHASE ---
    all_requests = []
    clause_blueprints = []

    for idx in successful_indices:
        meta = parent_metadata[idx]
        c1_signs, c2_signs = [], []
        
        # Pack surviving roots from Clause 1 (i.e. the given clause)
        for i, (sign, old_root) in enumerate(meta["c1"].literals):
            if i != meta["lit_idx1"]:
                all_requests.append([idx, old_root]) # [batch_idx, root_idx]
                c1_signs.append(sign)
        logger.debug(f"surviving roots from Claus 1 {all_requests}")

        # Pack surviving roots from Clause 2 (i.e. target clause)
        for i, (sign, old_root) in enumerate(meta["c2"].literals):
            if i != meta["lit_idx2"]:
                all_requests.append([idx, old_root]) # [batch_idx, root_idx]
                c2_signs.append(sign)
        logger.debug(f"surviving roots from Claus 2 {all_requests}")
                
        # Save the blueprint to re-stitch this clause later
        clause_blueprints.append({
            "c1_parent": meta["c1"],
            "c2_parent": meta["c2"],
            "c1_signs": c1_signs,
            "c2_signs": c2_signs
        })

    resolvents = []
    logger.debug(f"all requests {all_requests}")

    # debugging statment to see the nodes tensor
    arena_size_before = pipeline.parser.arena_ptr
    logger.debug(f"Arena Size BEFORE Instantiation: {arena_size_before} nodes")
    
    # --- 4. THE EXECUTION PHASE ---
    if all_requests:
        batched_requests_tensor = torch.tensor(all_requests, dtype=torch.long, device=pipeline.device)
        # 1 MASSIVE GPU CALL replaces the loop!
        new_roots_flat = pipeline.batched_instantiate_in_arena(batched_requests_tensor, unifier).tolist()
        logger.debug(f"new roots flat : {new_roots_flat}")
    else:
        # Handles the edge case where resolving two unit clauses leaves 0 roots to copy
        new_roots_flat = []

    # debugging 
    arena_size_after = pipeline.parser.arena_ptr
    nodes_minted = arena_size_after - arena_size_before
    logger.debug(f"Arena Size AFTER Instantiation: {arena_size_after} nodes")
    logger.debug(f"HIDDEN WORK: The GPU secretly minted {nodes_minted} new nodes to support roots {new_roots_flat}")
    if nodes_minted > 0:
        logger.debug(f"full nodes tensor: {pipeline.parser.nodes}")
        minted_nodes = pipeline.parser.nodes[arena_size_before:]
        minted_children = pipeline.parser.children[arena_size_before:]
        logger.debug(f"minuted nodes: {minted_nodes}")
        logger.debug(f"minted childre: {minted_children}")

        # map back to human readable symbols
        vocab_id_to_sym = { v: k for k, v in pipeline.parser.global_vocab.items()}
        vocab_id_to_sym[-2] = "VAR"

        human_symbols = [vocab_id_to_sym.get(n.item(), "?") for n in minted_nodes]
        logger.debug(f"Minted symbols : {human_symbols}")
        logger.debug(f"Minted pointers: \n{minted_children.tolist()}")

    # --- 5. THE RE-STITCHING PHASE ---
    ptr = 0
    for blueprint in clause_blueprints:
        new_literals = []
        
        # Pop exactly the right amount of roots for Clause 1
        num_c1 = len(blueprint["c1_signs"])
        for i in range(num_c1):
            new_literals.append((blueprint["c1_signs"][i], new_roots_flat[ptr]))
            ptr += 1
        logger.debug(f"new literals from clause 1: {new_literals}")
            
        # Pop exactly the right amount of roots for Clause 2
        num_c2 = len(blueprint["c2_signs"])
        for i in range(num_c2):
            new_literals.append((blueprint["c2_signs"][i], new_roots_flat[ptr]))
            ptr += 1
        logger.debug(f"new literals from clause 2: {new_literals}")
            
        new_clause = VirtualClause(new_literals)
        new_clause.deduplicate()
        new_clause.setDerivation(flatDerivation("resolution", [blueprint["c1_parent"], blueprint["c2_parent"]]))
        resolvents.append(new_clause)

        if logger.isEnabledFor(logging.INFO):
            readable_res = decode_virtual_clause(new_clause, pipeline)
            logger.info(f"New Resolvent: {readable_res}")

    return resolvents


def compute_given_clause_factors_tensor(given_clause, pipeline):
    factors = []
    queries = []
    metadata = []
    
    # --- 1. Find Intra-Clause Matches ---
    lits = given_clause.literals
    for i in range(len(lits)):
        for j in range(i + 1, len(lits)):
            sign1, root1 = lits[i]
            sign2, root2 = lits[j]
            
            if sign1 == sign2:
                queries.append([root1, root2])
                metadata.append({"lit_idx2": j}) # The literal we will drop
                
    if not queries:
        return factors
        
    # --- 2. Batched GPU Unification ---
    subs, success_mask, unifier = pipeline.prove_batch_indices(queries)
    
    successful_indices = torch.nonzero(success_mask).squeeze(-1)
    if successful_indices.dim() == 0:
        successful_indices = [successful_indices.item()]
    else:
        successful_indices = successful_indices.tolist()

    # --- 3. THE PACKING PHASE ---
    all_requests = []
    factor_blueprints = []

    for idx in successful_indices:
        meta = metadata[idx]
        signs = []
        
        # Pack surviving roots (skipping the redundant literal)
        for i, (sign, old_root) in enumerate(given_clause.literals):
            if i != meta["lit_idx2"]:
                all_requests.append([idx, old_root]) # [batch_idx, root_idx]
                signs.append(sign)
                
        factor_blueprints.append({"signs": signs})

    # --- 4. THE EXECUTION PHASE ---
    if all_requests:
        batched_requests_tensor = torch.tensor(all_requests, dtype=torch.long, device=pipeline.device)
        new_roots_flat = pipeline.batched_instantiate_in_arena(batched_requests_tensor, unifier).tolist()
    else:
        new_roots_flat = []

    # --- 5. THE RE-STITCHING PHASE ---
    ptr = 0
    for blueprint in factor_blueprints:
        new_literals = []
        num_signs = len(blueprint["signs"])
        
        # Pop the roots needed for this factor
        for i in range(num_signs):
            new_literals.append((blueprint["signs"][i], new_roots_flat[ptr]))
            ptr += 1
            
        new_clause = VirtualClause(new_literals)
        new_clause.deduplicate()
        new_clause.setDerivation(flatDerivation("factor", [given_clause]))
        factors.append(new_clause)
        
    return factors

def run_given_clause_benchmark_tensor(tptp_strings, pipeline, max_loops=2000):
    
    # 1. Parse all strings into the Python-list memory arena
    unprocessed = [parse_tptp_to_virtual_clause(s, pipeline) for s in tptp_strings]
    processed = []
    
    if isinstance(pipeline.parser.nodes, list):
        CAPACITY = 5_000_000  # VRAM Buffer Size (5 million nodes)
        initial_size = len(pipeline.parser.nodes)
        
        # Track the edge of the "live" data
        pipeline.parser.arena_ptr = initial_size 
        
        # Allocate massive empty tensors
        new_nodes = torch.zeros(CAPACITY, dtype=torch.long, device=pipeline.device)
        new_children = torch.full((CAPACITY, pipeline.parser.max_arity), -1, dtype=torch.long, device=pipeline.device)
        new_is_var = torch.zeros(CAPACITY, dtype=torch.bool, device=pipeline.device)
        
        # Copy the starting axioms into the top of the buffer
        new_nodes[:initial_size] = torch.tensor(pipeline.parser.nodes, dtype=torch.long, device=pipeline.device)
        new_children[:initial_size] = torch.tensor(pipeline.parser.children, dtype=torch.long, device=pipeline.device)
        new_is_var[:initial_size] = torch.tensor(pipeline.parser.is_var_mask, dtype=torch.bool, device=pipeline.device)
        
        # Overwrite the pipeline arrays
        pipeline.parser.nodes = new_nodes
        pipeline.parser.children = new_children
        pipeline.parser.is_var_mask = new_is_var
        
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