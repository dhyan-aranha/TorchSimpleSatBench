import torch
import logging
from TorchParse import LogicParser
from TorchUnify import BatchedGPUUnifier, ProverPipeline


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("TraceDemo")


class TracedBatchedGPUUnifier(BatchedGPUUnifier):
    
    def update_ref(self, indices, b_idx):
        logger.debug(f"[update_ref] START | Resolving {indices.shape[0]} pointers...")
        logger.debug(f"[update_ref] Input Indices: {indices.tolist()}")
        logger.debug(f"[update_ref] Input Batches: {b_idx.tolist()}")
        
        curr = indices.clone()
        valid_mask = (curr != -1)
        active = valid_mask.clone()
        
        loop_count = 0
        while active.any():
            loop_count += 1
            nxt = curr.clone()
            nxt[active] = self.subs[b_idx[active], curr[active]]
            
            changed = (nxt != curr) & active
            curr = nxt
            active = changed
            logger.debug(f"  -> [update_ref] Iteration {loop_count}: Current pointers at {curr.tolist()}")
            
        logger.debug(f"[update_ref] DONE | Resolved to: {curr.tolist()}")
        return curr
    
    def occurs_check(self, var_indices, target_indices, b_idx):
        logger.debug(f"[occurs_check] START | Checking {var_indices.shape[0]} bindings for cycles.")
        logger.debug(f"[occurs_check] Variables: {var_indices.tolist()} | Targets: {target_indices.tolist()} | Batches: {b_idx.tolist()}")
        
        K = var_indices.shape[0]
        if K == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.nodes.device)

        device = self.nodes.device
        failed_occurs = torch.zeros(K, dtype=torch.bool, device=device)
        frontier = target_indices.unsqueeze(1) 
        
        
        wave = 0
        while frontier.shape[1] > 0:
            wave += 1
            logger.debug(f"occurs_check frontier : {frontier}")
            logger.debug(f"  -> [occurs_check] Wave {wave} | Frontier Shape: {frontier.shape}")
            
            flat_frontier = frontier.flatten()
            flat_b_idx = b_idx.unsqueeze(1).expand(-1, frontier.shape[1]).flatten()
        
            
            flat_updated = self.update_ref(flat_frontier, flat_b_idx)
            frontier = flat_updated.view(K, -1)
            logger.debug(f"[occurs_check] Updated frontier : {frontier}")
            
            matches = (frontier == var_indices.unsqueeze(1))
            new_fails = matches.any(dim=1)
            failed_occurs = failed_occurs | new_fails
            
            if new_fails.any():
                logger.warning(f"  -> [occurs_check] CYCLE DETECTED in current wave")
            
            active_rows = ~failed_occurs
            if not active_rows.any():
                break
                
            valid_mask = (frontier != -1)
            safe_frontier = frontier.clone()
            safe_frontier[~valid_mask] = 0 
            
            next_frontier = self.children[safe_frontier]
            logger.debug(f"next frontier raw : {next_frontier}") 
            next_frontier[~valid_mask] = -1
            frontier = next_frontier.view(K, -1)
            logger.debug(f"next frontier : {frontier}")
            
            col_has_data = (frontier != -1).any(dim=0)
            frontier = frontier[:, col_has_data]

        logger.debug(f"[occurs_check] DONE | Failed mask: {failed_occurs.tolist()}")
        return failed_occurs

    
    def unify(self, pair_batch):
        logger.info(f"\n{'='*60}\n[unify] Starting unification...\n{'='*60}")
        
       
        logger.info("--- Parsed representation of AST ---")
        logger.info(f"Nodes (Vocab IDs): {self.nodes.tolist()}")
        logger.info(f"Is_Var Mask:       {self.is_var_mask.tolist()}")
        logger.info(f"Children Matrix: \n{self.children}")
        logger.info(f"Roots : {pair_batch} ")
        logger.info("---------------------------")
        
        num_pairs = pair_batch.shape[0]

        success_mask = torch.ones(num_pairs, dtype=torch.bool, device=self.nodes.device)
        batch_idx = torch.arange(num_pairs, dtype=torch.long, device=self.nodes.device)
        
        base_subs = torch.arange(self.num_nodes, dtype=torch.long, device=self.nodes.device)
        self.subs = base_subs.unsqueeze(0).expand(num_pairs, -1).clone()
        
        
        logger.info(f" Initial subs tensor (Shape {self.subs.shape}):\n{self.subs}")
        
        frontier = pair_batch
        wave = 0
        
        while frontier.shape[0] > 0:
            wave += 1
            logger.info(f"\n--- [unify] WAVE {wave} ---")
            
            alive_mask = success_mask[batch_idx]
            frontier = frontier[alive_mask]
            batch_idx = batch_idx[alive_mask]
            
            if frontier.shape[0] == 0: 
                logger.debug("All active branches terminated. Exiting loop.")
                break
            
            logger.debug(f"Active Frontier Pairs: {frontier.tolist()}")
            logger.debug(f"Corresponding Batches: {batch_idx}")
            
            left = self.update_ref(frontier[:, 0], batch_idx)
            right = self.update_ref(frontier[:, 1], batch_idx)
            
            active = (left != right)

            num_trivial = (~active).sum().item()
            if num_trivial > 0:
                trivial_left = left[~active].tolist()
                logger.debug(f"Dropped {num_trivial} pairs due to trival equality (left == right) : {trivial_left}")

            left, right = left[active], right[active]
            batch_idx = batch_idx[active] 
            
            if left.shape[0] == 0: 
                logger.debug("All pairs resolved to identical pointers. Trivial success.")
                break
            
            l_is_v = self.is_var_mask[left]
            r_is_v = self.is_var_mask[right]
            
            next_frontier_pieces = []
            next_batch_pieces = []
            
            # --- PROCESS VARIABLES ---
            is_var_pair = l_is_v | r_is_v
            logger.debug(f"Identified {is_var_pair.sum().item()} variable binding equations: {is_var_pair}.")
            
            if torch.any(is_var_pair):
                v_left = left[is_var_pair]
                v_right = right[is_var_pair]
                v_l_is_v = l_is_v[is_var_pair]
                
                v_idx = torch.where(v_l_is_v, v_left, v_right)
                t_idx = torch.where(v_l_is_v, v_right, v_left)
                b_idx_all = batch_idx[is_var_pair]
                
                # Race Condition Deferral Logic
                unique_b, inverse_indices = torch.unique(b_idx_all, return_inverse=True)
                idx_seq = torch.arange(len(b_idx_all), device=self.nodes.device)
                
                first_occ_idx = torch.zeros(len(unique_b), dtype=torch.long, device=self.nodes.device)
                first_occ_idx.scatter_(0, inverse_indices.flip(0), idx_seq.flip(0))
                
                process_v = v_idx[first_occ_idx]
                process_t = t_idx[first_occ_idx]
                process_b = b_idx_all[first_occ_idx]
                
                deferred_mask = torch.ones(len(b_idx_all), dtype=torch.bool, device=self.nodes.device)
                deferred_mask[first_occ_idx] = False
                
                if torch.any(deferred_mask):
                    logger.info(f"RACE CONDITION PREVENTED: Deferred {deferred_mask.sum().item()} equations to next wave : {[v_left[deferred_mask], v_right[deferred_mask]]}.")
                    next_frontier_pieces.append(torch.stack([v_left[deferred_mask], v_right[deferred_mask]], dim=1))
                    next_batch_pieces.append(b_idx_all[deferred_mask])
                    
                failed_occurs = self.occurs_check(process_v, process_t, process_b)
                
                if torch.any(failed_occurs):
                    logger.warning(f"Killing batches due to occurs check: {process_b[failed_occurs].tolist()}")
                    success_mask[process_b[failed_occurs]] = False
                    
                survivors = ~failed_occurs
                s_v = process_v[survivors]
                s_t = process_t[survivors]
                s_b = process_b[survivors]
                
                if s_v.shape[0] > 0:
                    logger.debug(f"Writing Substitutions: Variables {s_v.tolist()} -> Targets {s_t.tolist()} (Batches {s_b.tolist()})")
                    self.subs[s_b, s_v] = s_t
                    # --- ADDED: Log the mutated substitution matrix ---
                    logger.info(f"EVOLVED SUBS TENSOR (Wave {wave}):\n{self.subs}")
                
            # --- PROCESS FUNCTIONS ---
            fun_mask = ~is_var_pair
            logger.debug(f"Identified {fun_mask.sum().item()} function/constant structure pairs.")
            
            if torch.any(fun_mask):
                f_left, f_right = left[fun_mask], right[fun_mask]
                f_batch = batch_idx[fun_mask]
                
                mismatch = (self.nodes[f_left] != self.nodes[f_right])
                if torch.any(mismatch):
                    logger.warning(f"Killing batches due to Structural Symbol Clash: {f_batch[mismatch].tolist()}")
                    success_mask[f_batch[mismatch]] = False
                
                valid_struct = ~mismatch
                f_left, f_right = f_left[valid_struct], f_right[valid_struct]
                f_batch = f_batch[valid_struct]
                
                if f_left.shape[0] > 0:
                    logger.debug("Extracting children from matched structures to queue for next wave.")
                    c_left = self.children[f_left]
                    c_right = self.children[f_right]
                    
                    c_batch = f_batch.unsqueeze(1).expand(-1, self.max_arity).flatten()
                    c_left, c_right = c_left.flatten(), c_right.flatten()
                    
                    valid_pad = (c_left != -1) & (c_right != -1)
                    
                    if torch.any(valid_pad):
                        next_frontier_pieces.append(torch.stack([c_left[valid_pad], c_right[valid_pad]], dim=1))
                        next_batch_pieces.append(c_batch[valid_pad])
            
            if len(next_frontier_pieces) > 0:
                frontier = torch.cat(next_frontier_pieces, dim=0)
                batch_idx = torch.cat(next_batch_pieces, dim=0)
                logger.debug(f"End of wave {wave}: Queued {frontier.shape[0]} new children for then next wave.")
            else:
                frontier = torch.empty((0, 2), dtype=torch.long, device=self.nodes.device)
                logger.debug(f"End of wave {wave}: no structural children to evaluate. Frontier is empty!")

        logger.info(f"\n[unify] DONE | Success Mask: {success_mask.tolist()}")
        return self.subs, success_mask


class TracedProverPipeline(ProverPipeline):
    
    def prove_batch_indices(self, pair_indices, standardize_apart=True):

        self.last_run_standardized = standardize_apart
        if not pair_indices:
            return torch.empty(0), torch.empty(0), None

        pair_batch = torch.tensor(pair_indices, dtype=torch.long, device=self.device)
        
        unifier = TracedBatchedGPUUnifier(
            self.parser.nodes, 
            self.parser.children, 
            self.parser.is_var_mask, 
            max_arity=self.parser.max_arity
        )
        
        subs, success_mask = unifier.unify(pair_batch)
        return subs, success_mask, unifier
    
    def instantiate(self, batched_requests, unifier):
        """
        A stripped-down instantiation purely for benchmarking.
        It computes the substituted DAGs and allocates them in VRAM, 
        but DOES NOT append them to the global arena.
        """

        if batched_requests.numel() == 0 or batched_requests.shape[0] == 0:
            logger.debug("No successful unifications to instantiate. Exiting graph copy early.")
            return torch.empty(0, dtype=torch.long, device=self.device)

        current_arena_size = self.parser.nodes.shape[0]
        
        original_b_idx = batched_requests[:, 0]
        roots_old = batched_requests[:, 1]

        logger.debug(f"Input Requests (Batch_ID, Root_ID): \n{batched_requests.tolist()}")
        
        unique_batches, local_b_idx = torch.unique(original_b_idx, return_inverse=True)
        num_unique_batches = unique_batches.shape[0]
        
        old_to_new = torch.full(
            (num_unique_batches * current_arena_size,), -1, 
            dtype=torch.long, device=self.device
        )
        
        # Apply root substitutions
        true_roots = unifier.update_ref(roots_old, original_b_idx)
        root_keys = (local_b_idx * current_arena_size) + true_roots
        
        unique_root_keys = torch.unique(root_keys)
        num_new_roots = unique_root_keys.numel()
        
        new_ids = torch.arange(current_arena_size, current_arena_size + num_new_roots, device=self.device)
        old_to_new[unique_root_keys] = new_ids
        next_alloc_idx = current_arena_size + num_new_roots
        
        frontier_keys = unique_root_keys
        out_nodes, out_is_var, out_children = [], [], []
        
        # Traverse and build the new subgraphs
        while frontier_keys.numel() > 0:
            current_local_b_idx = torch.div(frontier_keys, current_arena_size, rounding_mode='floor')
            current_nodes = frontier_keys % current_arena_size
            
            level_nodes = self.parser.nodes[current_nodes]
            level_is_var = self.parser.is_var_mask[current_nodes]
            level_children = self.parser.children[current_nodes] 
            
            valid_mask = level_children != -1
            
            if valid_mask.any():
                expanded_local_b_idx = current_local_b_idx.unsqueeze(1).expand(-1, self.parser.max_arity)
                valid_local_b_idx = expanded_local_b_idx[valid_mask]
                valid_children_old = level_children[valid_mask]
                
                valid_original_b_idx = unique_batches[valid_local_b_idx]
                
                # Query the substitution matrix
                true_children = unifier.update_ref(valid_children_old, valid_original_b_idx)
                
                child_keys = (valid_local_b_idx * current_arena_size) + true_children
                unallocated_mask = old_to_new[child_keys] == -1
                unallocated_keys = child_keys[unallocated_mask]
                
                unique_new_keys = torch.unique(unallocated_keys)
                num_new_children = unique_new_keys.numel()
                
                if num_new_children > 0:
                    new_child_ids = torch.arange(next_alloc_idx, next_alloc_idx + num_new_children, device=self.device)
                    old_to_new[unique_new_keys] = new_child_ids
                    next_alloc_idx += num_new_children
                    
                new_level_children = torch.full_like(level_children, -1)
                new_level_children[valid_mask] = old_to_new[child_keys]
                frontier_keys = unique_new_keys
            else:
                new_level_children = torch.full_like(level_children, -1)
                frontier_keys = torch.empty(0, dtype=torch.long, device=self.device)
                
            out_nodes.append(level_nodes)
            out_is_var.append(level_is_var)
            out_children.append(new_level_children)
            
        if out_nodes:
            ephemeral_nodes = torch.cat(out_nodes)
            ephemeral_is_var = torch.cat(out_is_var)
            ephemeral_children = torch.cat(out_children)
            
            
            return old_to_new[root_keys], ephemeral_nodes, ephemeral_is_var, ephemeral_children
            
        return old_to_new[root_keys], None, None, None

    def batched_instantiate_in_arena(self, batched_requests, unifier):
        logger.info(f"\n{'='*60}\n[batched_instantiate] STARTING MEMORY ARENA GRAPH COPY\n{'='*60}")

        if batched_requests.numel() == 0 or batched_requests.shape[0] == 0:
            logger.debug("No successful unifications to instantiate. Exiting graph copy early.")
            return torch.empty(0, dtype=torch.long, device=self.device)

        current_arena_size = self.parser.nodes.shape[0]
        original_b_idx = batched_requests[:, 0]
        roots_old = batched_requests[:, 1]
        
        logger.debug(f"Input Requests (Batch_ID, Root_ID): \n{batched_requests.tolist()}")
        
        unique_batches, local_b_idx = torch.unique(original_b_idx, return_inverse=True)
        num_unique_batches = unique_batches.shape[0]
        logger.debug(f"Reality Compression: {original_b_idx.tolist()} -> mapped to dense local batches {local_b_idx.tolist()}")
        
        old_to_new = torch.full(
            (num_unique_batches * current_arena_size,), -1, 
            dtype=torch.long, device=self.device
        )
        
        true_roots = unifier.update_ref(roots_old, original_b_idx)
        root_keys = (local_b_idx * current_arena_size) + true_roots
        
        unique_root_keys = torch.unique(root_keys)
        num_new_roots = unique_root_keys.numel()
        
        new_ids = torch.arange(current_arena_size, current_arena_size + num_new_roots, device=self.device)
        old_to_new[unique_root_keys] = new_ids
        next_alloc_idx = current_arena_size + num_new_roots
        
        logger.debug(f"Minted {num_new_roots} unique initial roots. New VRAM Arena: {next_alloc_idx}")
        
        frontier_keys = unique_root_keys
        out_nodes, out_is_var, out_children = [], [], []
        
        wave = 0
        while frontier_keys.numel() > 0:
            wave += 1
            logger.info(f"\n--- [batched_instantiate] BFS WAVE {wave} ---")
            logger.debug(f"Processing {frontier_keys.numel()} unique nodes in this depth layer.")

            current_local_b_idx = torch.div(frontier_keys, current_arena_size, rounding_mode='floor')
            current_nodes = frontier_keys % current_arena_size
            
            level_nodes = self.parser.nodes[current_nodes]
            level_is_var = self.parser.is_var_mask[current_nodes]
            level_children = self.parser.children[current_nodes] 
            
            valid_mask = level_children != -1
            
            if valid_mask.any():
                expanded_local_b_idx = current_local_b_idx.unsqueeze(1).expand(-1, self.parser.max_arity)
                valid_local_b_idx = expanded_local_b_idx[valid_mask]
                valid_children_old = level_children[valid_mask]
                
                valid_original_b_idx = unique_batches[valid_local_b_idx]
                
                logger.debug(f"Querying Update_Ref for {valid_children_old.numel()} structural children...")
                true_children = unifier.update_ref(valid_children_old, valid_original_b_idx)
                
                child_keys = (valid_local_b_idx * current_arena_size) + true_children
                
                unallocated_mask = old_to_new[child_keys] == -1
                unallocated_keys = child_keys[unallocated_mask]
                
                unique_new_keys = torch.unique(unallocated_keys)
                num_new_children = unique_new_keys.numel()
                
                if num_new_children > 0:
                    logger.debug(f"Memoization Miss: Minting {num_new_children} new distinct children...")
                    new_child_ids = torch.arange(next_alloc_idx, next_alloc_idx + num_new_children, device=self.device)
                    old_to_new[unique_new_keys] = new_child_ids
                    next_alloc_idx += num_new_children
                else:
                    logger.debug("Memoization Hit: All children were already allocated in previous waves. Perfect Graph sharing!")
                    
                new_level_children = torch.full_like(level_children, -1)
                new_level_children[valid_mask] = old_to_new[child_keys]
                
                frontier_keys = unique_new_keys
            else:
                new_level_children = torch.full_like(level_children, -1)
                frontier_keys = torch.empty(0, dtype=torch.long, device=self.device)
                
            # Visualize the generated layer
            logger.info(f"  -> BUILT LAYER {wave}:")
            logger.info(f"     Nodes (Vocab IDs) : {level_nodes.tolist()}")
            logger.info(f"     Children Wiring   :\n{new_level_children.tolist()}")
            

            out_nodes.append(level_nodes)
            out_is_var.append(level_is_var)
            out_children.append(new_level_children)
            
        if out_nodes:
            logger.info("\n[batched_instantiate] Appending instantiated graphs to global VRAM Arena...")
            self.parser.nodes = torch.cat([self.parser.nodes] + out_nodes)
            self.parser.is_var_mask = torch.cat([self.parser.is_var_mask] + out_is_var)
            self.parser.children = torch.cat([self.parser.children] + out_children)
            
            # Visualize the final global state update
            total_added = sum(n.shape[0] for n in out_nodes)
            logger.info(f"Successfully appended {total_added} total nodes to the global arena.")
            
        print(f"Nodes : {self.parser.nodes}")
        print(f"Children : {self.parser.children}")           
        return old_to_new[root_keys]
    
    def decode_term(self, idx, unifier, id_to_symbol, var_name_map, visited=None, batch_idx=0, depth=0):
        indent = "  " * depth
        if visited is None:
            visited = set()
            
        idx_int = int(idx) if isinstance(idx, torch.Tensor) else int(idx)
        logger.debug(f"{indent}[decode_term] Reading physical node at memory index: {idx_int}")
        
        node_tensor = torch.tensor([idx_int], dtype=torch.long, device=self.device)
        logger.debug(f"[decode_term] The idx_int tensor i.e. node_tensor is {node_tensor}")
        b_idx_tensor = torch.tensor([batch_idx], dtype=torch.long, device=self.device)
        logger.debug(f"[decode_term] The b_idx_tensor is : {b_idx_tensor}")
        
        true_idx = unifier.update_ref(node_tensor, b_idx_tensor).item()
        
        if true_idx != idx_int:
            logger.debug(f"{indent} -> Pointer followed! Node {idx_int} redirects to Node {true_idx}")
            
        if true_idx in visited:
            logger.warning(f"{indent} -> [CYCLE DETECTED] at Node {true_idx}")
            return "[CYCLE DETECTED]"
        visited.add(true_idx)
        
        if unifier.is_var_mask[true_idx]:
            # It's a variable! Let's find its human name.
            logger.debug(f"{indent} -> Node {true_idx} is a VARIABLE.")
            idx_to_name = {int(v): k for k, v in var_name_map.items()}
            
            if true_idx in idx_to_name:
                human_name = idx_to_name[true_idx]
                logger.debug(f"{indent} -> Lookup SUCCESS: Memory {true_idx} maps to Original String '{human_name}'")
                return human_name
                
            var_letters = ['X', 'Y', 'Z', 'V', 'W', 'U']
            assigned_letter = var_letters[true_idx % len(var_letters)]
            fallback_name = f"{assigned_letter}_{true_idx}"
            logger.debug(f"{indent} -> Lookup FAILED: No original string for {true_idx}. Using fallback '{fallback_name}'")
            return fallback_name
            
        # It's a function or constant
        sym_id = unifier.nodes[true_idx].item()
        sym_str = id_to_symbol.get(sym_id, "?")
        logger.debug(f"{indent} -> Node {true_idx} is a FUNCTION/CONSTANT. Vocab ID: {sym_id} -> String: '{sym_str}'")
        
        children = unifier.children[true_idx]
        valid_children = children[children != -1]
        
        if len(valid_children) == 0:
            logger.debug(f"{indent} -> '{sym_str}' has no children. Returning as constant.")
            return sym_str
        else:
            logger.debug(f"{indent} -> '{sym_str}' has {len(valid_children)} children. Diving deeper...")
            args = []
            for i, c in enumerate(valid_children):
                logger.debug(f"{indent} --- Decoding Argument {i+1} of '{sym_str}' ---")
                args.append(self.decode_term(c.item(), unifier, id_to_symbol, var_name_map, visited.copy(), batch_idx, depth + 1))
            
            reconstructed = f"{sym_str}({', '.join(args)})"
            logger.debug(f"{indent} -> Assembled Sub-Tree: {reconstructed}")
            return reconstructed
        
    def print_report(self, string_pairs, subs, success_mask, unifier):
        print("\n" + "="*60)
        print(f"Batch Unification Log")
        print("="*60)
        
        id_to_symbol = {v: k for k, v in self.parser.global_vocab.items()}
        offset = getattr(self, 'batch_var_map_offset', 0)
        
        for i, (left_str, right_str) in enumerate(string_pairs):
            status = "SUCCESS" if success_mask[i].item() else "FAILED"
            print(f"\nPair {i}: {left_str}  <=>  {right_str}")
            print(f"Status: {status}")
            
            if success_mask[i].item():
                if not self.last_run_standardized:
                    combined_var_map = self.parser.var_maps[offset + i]
                else:
                    left_map = self.parser.var_maps[offset + i * 2]
                    right_map = self.parser.var_maps[offset + i * 2 + 1]
                    combined_var_map = {**left_map}
                    for var_name, idx in right_map.items():
                        combined_var_map[f"{var_name}_2"] = idx
                
                if not combined_var_map:
                    print("   -> No variables to bind.")
                else:
                    print("   -> Bindings:")
                    for var_name, var_memory_idx in combined_var_map.items():
                        bound_string = self.decode_term(var_memory_idx, unifier, id_to_symbol, combined_var_map, batch_idx=i)
                        if bound_string != var_name:
                            print(f"      {var_name} = {bound_string}")
            else:
                print("   -> Rejected due to Symbol Mismatch or Occurs Check.")
        

if __name__ == "__main__":
    
    string_pairs = [
        #("X", "a"),
        #("X", "f(X)"),
        #("X", "f(Y)"),
        ("f(g(X,Y), h(a, b))", "f(g(X, a)), h(Y, X))"),
        #("f(X, g(a))", "f(X, Y)"), 
        #("f(X, g(a))", "f(X, X)"), 
        #("g(X)", "g(f(g(X),b))"),
        #("p(X,X,X)", "p(Y,Y,e)"),
        #("f(f(g(X),a),X)", "f(Y,g(Y))"),
        #("f(f(g(X),a),g(X))", "f(Y,g(Z))"),
        #("p(X,g(a), f(a, f(a)))", "p(f(a), g(Y), f(Y, Z))")
    ]
    
    logger.info("Initializing Traced Pipeline...")
    pipeline = TracedProverPipeline()
    
    nodes, children, is_var, root_pairs = pipeline.parser.parse_pairs(string_pairs)
    
    if isinstance(pipeline.parser.nodes, list):
        pipeline.parser.nodes = torch.tensor(pipeline.parser.nodes, dtype=torch.long, device=pipeline.device)
        pipeline.parser.children = torch.tensor(pipeline.parser.children, dtype=torch.long, device=pipeline.device)
        pipeline.parser.is_var_mask = torch.tensor(pipeline.parser.is_var_mask, dtype=torch.bool, device=pipeline.device)
    
    subs, success_mask, unifier = pipeline.prove_batch_indices(root_pairs.tolist(), standardize_apart=False)
    
    successful_indices = torch.nonzero(success_mask).squeeze(-1)
    if successful_indices.dim() == 0:
        successful_indices = successful_indices.unsqueeze(0)
    
    batched_requests = []
    for idx in successful_indices.tolist():
        left_root = root_pairs[idx][0].item()
        batched_requests.append([idx, left_root])
    
    batched_requests_tensor = torch.tensor(batched_requests, dtype=torch.long)
    new_roots = pipeline.batched_instantiate_in_arena(batched_requests_tensor, unifier)
    
    pipeline.print_report(string_pairs, subs, success_mask, unifier)

    logger.info(f"\n{'='*60}\n [Translation back to strings] \n{'='*60}")
    
    logger.debug("Creating a 'Dummy Unifier' (Clean Identity Matrix).")
    logger.debug("Because the pointers are now physically wired together in the VRAM arena, we DO NOT want to apply the old substitutions. We just read the physical edges directly.")
    dummy_unifier = BatchedGPUUnifier(
        pipeline.parser.nodes, 
        pipeline.parser.children, 
        pipeline.parser.is_var_mask, 
        pipeline.parser.max_arity
    )

    new_roots_list = new_roots.tolist()
    logger.debug(f"New roots {new_roots_list}")
    id_to_symbol = {v: k for k, v in pipeline.parser.global_vocab.items()}
    logger.debug(f"global dictionary: {id_to_symbol}")
    offset = getattr(pipeline, 'batch_var_map_offset', 0)
    logger.debug(f" The offset : {offset}")

    for i, orig_batch_idx in enumerate(successful_indices.tolist()):
        new_root = new_roots_list[i]
        left_str, right_str = string_pairs[orig_batch_idx]
        
        logger.info(f"\n--- Decoding Result for Pair {orig_batch_idx}: {left_str} <=> {right_str} ---")
        logger.debug(f"Target Root Index to decode: {new_root}")
        
        if not pipeline.last_run_standardized:
            logger.debug(f"{pipeline.parser.var_maps}")
            combined_var_map = pipeline.parser.var_maps[offset + orig_batch_idx]
            logger.debug(f"Translation Dictionary (Single Clause): {combined_var_map}")
        else:
            left_map = pipeline.parser.var_maps[offset + orig_batch_idx * 2]
            right_map = pipeline.parser.var_maps[offset + orig_batch_idx * 2 + 1]
            combined_var_map = {**left_map}
            for var_name, v_idx in right_map.items():
                combined_var_map[f"{var_name}_2"] = v_idx
            logger.debug(f"Translation Dictionary (Merged Clauses): {combined_var_map}")
                
        logger.debug("Initiating Graph Traversal...")
        
        # Because we use the Dummy Unifier, batch_idx=0 is perfectly safe here!
        instantiated_string = pipeline.decode_term(
            new_root, dummy_unifier, id_to_symbol, combined_var_map, batch_idx=0
        )
        
        logger.info(f">>> FINAL RESULT: {instantiated_string} <<<\n")