import torch
import math
from TorchParse import LogicParser

# There are two important classes here: BatchedGPUUnifier and ProverPipeline. BatchedGPUUnifier, 
# takes as input a batch of pairs of formulas and computes the most general unifier of each pair in the
# the batch. ProverPiple does three things the most important of which is subsitution: 
# 1) pre-processing : prove_batch_indices passes the parsed batch of pairs of formulas to BatchedGPUUnifier
# 2) subsitution : batched_instantiate_in_arena this is the by far the most complicated part of the code.
#    given a batch of unifications and a batch of formulas, attempts to apply to substitute the mgu's computed by 
#    BatchedGPUUnifier to the formulas to outputs their representation as pytorch tensors. 
# 3) post-processing : decode_term converts our pytorch tensor representation back to strings when we wish to print
#    the final human readable answer. 

class BatchedGPUUnifier:
    def __init__(self, nodes, children, is_var_mask, max_arity):
        self.nodes = nodes
        self.children = children
        self.is_var_mask = is_var_mask
        self.num_nodes = nodes.shape[0]
        self.max_arity = max_arity
        self.subs = torch.arange(self.num_nodes, dtype=torch.long, device=nodes.device).unsqueeze(0)

    def update_ref(self, indices, b_idx):
        """
        Follows the substitution chain to find the true root.
        Reads strictly from the 2D substitution matrix using b_idx.
        """
        curr = indices.clone()
        valid_mask = (curr != -1)
        active = valid_mask.clone()
        
        while active.any():
            nxt = curr.clone()

            nxt[active] = self.subs[b_idx[active], curr[active]]
            
            changed = (nxt != curr) & active
            curr = nxt
            active = changed
            
        return curr
    
    def occurs_check(self, var_indices, target_indices, b_idx):
        """
        Dynamically bounds the Breadth-First Search.
        Now safely threads the batch index down into update_ref.
        """
        K = var_indices.shape[0]
        if K == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.nodes.device)

        device = self.nodes.device
        failed_occurs = torch.zeros(K, dtype=torch.bool, device=device)
        
        # Initial shape: [K, 1]
        frontier = target_indices.unsqueeze(1) 
        
        while frontier.shape[1] > 0:
            
            flat_frontier = frontier.flatten()
            flat_b_idx = b_idx.unsqueeze(1).expand(-1, frontier.shape[1]).flatten()
            
            flat_updated = self.update_ref(flat_frontier, flat_b_idx)
            frontier = flat_updated.view(K, -1)
            
            matches = (frontier == var_indices.unsqueeze(1))
            new_fails = matches.any(dim=1)
            failed_occurs = failed_occurs | new_fails
            
            active_rows = ~failed_occurs
            if not active_rows.any():
                break
                
            valid_mask = (frontier != -1)
            safe_frontier = frontier.clone()
            safe_frontier[~valid_mask] = 0 
            
            next_frontier = self.children[safe_frontier] 
            next_frontier[~valid_mask] = -1
            
            frontier = next_frontier.view(K, -1)
            
            col_has_data = (frontier != -1).any(dim=0)
            frontier = frontier[:, col_has_data]

        return failed_occurs

    def unify(self, pair_batch):
        """
        Iterative, batched unification loop.
        Allocates a clean 2D [num_pairs, num_nodes] substitution matrix per run.
        """
        num_pairs = pair_batch.shape[0]

        success_mask = torch.ones(num_pairs, dtype=torch.bool, device=self.nodes.device)
        batch_idx = torch.arange(num_pairs, dtype=torch.long, device=self.nodes.device)
        
        # Initialize the 2D substitution matrix
        base_subs = torch.arange(self.num_nodes, dtype=torch.long, device=self.nodes.device)
        self.subs = base_subs.unsqueeze(0).expand(num_pairs, -1).clone()
        
        frontier = pair_batch
        
        while frontier.shape[0] > 0:
            alive_mask = success_mask[batch_idx]
            frontier = frontier[alive_mask]
            batch_idx = batch_idx[alive_mask]
            
            if frontier.shape[0] == 0: break
            
            
            left = self.update_ref(frontier[:, 0], batch_idx)
            right = self.update_ref(frontier[:, 1], batch_idx)
            
            active = (left != right)
            left, right = left[active], right[active]
            batch_idx = batch_idx[active] 
            
            if left.shape[0] == 0: break
            
            l_is_v = self.is_var_mask[left]
            r_is_v = self.is_var_mask[right]
            
            next_frontier_pieces = []
            next_batch_pieces = []
            
            
            is_var_pair = l_is_v | r_is_v
            
            if torch.any(is_var_pair):
                v_left = left[is_var_pair]
                v_right = right[is_var_pair]
                v_l_is_v = l_is_v[is_var_pair]
                
                
                v_idx = torch.where(v_l_is_v, v_left, v_right)
                t_idx = torch.where(v_l_is_v, v_right, v_left)
                b_idx_all = batch_idx[is_var_pair]
                
                
                
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
                    next_frontier_pieces.append(torch.stack([v_left[deferred_mask], v_right[deferred_mask]], dim=1))
                    next_batch_pieces.append(b_idx_all[deferred_mask])
                    
                # OCCURS CHECK
                failed_occurs = self.occurs_check(process_v, process_t, process_b)
                
                if torch.any(failed_occurs):
                    success_mask[process_b[failed_occurs]] = False
                    
                survivors = ~failed_occurs
                s_v = process_v[survivors]
                s_t = process_t[survivors]
                s_b = process_b[survivors]
                
                
                if s_v.shape[0] > 0:
                    self.subs[s_b, s_v] = s_t
                
            fun_mask = ~is_var_pair
            if torch.any(fun_mask):
                f_left, f_right = left[fun_mask], right[fun_mask]
                f_batch = batch_idx[fun_mask]
                
                mismatch = (self.nodes[f_left] != self.nodes[f_right])
                if torch.any(mismatch):
                    success_mask[f_batch[mismatch]] = False
                
                valid_struct = ~mismatch
                f_left, f_right = f_left[valid_struct], f_right[valid_struct]
                f_batch = f_batch[valid_struct]
                
                if f_left.shape[0] > 0:
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
            else:
                frontier = torch.empty((0, 2), dtype=torch.long, device=self.nodes.device)

        return self.subs, success_mask
    
# Auxiliary class 
class SingleUnifierWrapper:
    def __init__(self, subs_row):
        self.subs = subs_row
        
    def update_ref(self, indices, b_idx=None): # Safe default handles both signatures
        curr = indices.clone()
        valid_mask = (curr != -1)
        active = valid_mask.clone()
        while active.any():
            nxt = curr.clone()
            nxt[active] = self.subs[curr[active]]
            changed = (nxt != curr) & active
            curr = nxt
            active = changed
        return curr

class ProverPipeline:
    """
    This class includes both pre- and pos-processing for our batches of claues. 
    """
    def __init__(self, device='cpu', max_arity=0):
        self.device = device

        self.parser = LogicParser()
        if max_arity > 0:
            self.parser.max_arity = max_arity

    
    def prove_batch_indices(self, pair_indices, standardize_apart=True):
        """
        Executes batched unification using purely integer memory pointers.
        Bypasses string parsing entirely for maximum performance.
        
        pair_indices: A list of tuples like [(root_idx_1, root_idx_2), ...]
        """
        if not pair_indices:
            return torch.empty(0), torch.empty(0), None

        # Convert the list of tuples into a [Batch_Size, 2] PyTorch tensor
        pair_batch = torch.tensor(pair_indices, dtype=torch.long, device=self.device)
        
        unifier = BatchedGPUUnifier(
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
            return torch.empty(0, dtype=torch.long, device=self.device)

        # We still need to know where the global arena ends so our new 
        # pointers don't accidentally overwrite existing axioms.
        current_arena_size = self.parser.nodes.shape[0]
        
        original_b_idx = batched_requests[:, 0]
        roots_old = batched_requests[:, 1]
        
        unique_batches, local_b_idx = torch.unique(original_b_idx, return_inverse=True)
        num_unique_batches = unique_batches.shape[0]
        
        old_to_new = torch.full(
            (num_unique_batches * current_arena_size,), -1, 
            dtype=torch.long, device=self.device
        )
        
        # 1. Apply root substitutions
        true_roots = unifier.update_ref(roots_old, original_b_idx)
        root_keys = (local_b_idx * current_arena_size) + true_roots
        
        unique_root_keys = torch.unique(root_keys)
        num_new_roots = unique_root_keys.numel()
        
        new_ids = torch.arange(current_arena_size, current_arena_size + num_new_roots, device=self.device)
        old_to_new[unique_root_keys] = new_ids
        next_alloc_idx = current_arena_size + num_new_roots
        
        frontier_keys = unique_root_keys
        out_nodes, out_is_var, out_children = [], [], []
        
        # 2. Traverse and build the new subgraphs
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
        """
        batched_requests: A 2D tensor of shape [R, 2] where R is the number of SUCCESSFUL requests. 
                          Column 0 is the original batch_idx (reality ID).
                          Column 1 is the root_idx to copy.
        """

        if batched_requests.numel() == 0 or batched_requests.shape[0] == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)

        current_arena_size = self.parser.nodes.shape[0]
        
        original_b_idx = batched_requests[:, 0]
        roots_old = batched_requests[:, 1]
        
        
        # Compress the true batch IDs into dense local coordinates [0, 1, 2...]
        # This ensures all literals from the same batch share the same memoization layer.
        unique_batches, local_b_idx = torch.unique(original_b_idx, return_inverse=True)
        num_unique_batches = unique_batches.shape[0]
        
        # 2D memoization table scaled by unique realities, not request rows
        old_to_new = torch.full(
            (num_unique_batches * current_arena_size,), -1, 
            dtype=torch.long, device=self.device
        )
        
        # Update the reference of the roots
        true_roots = unifier.update_ref(roots_old, original_b_idx)
        
        # Generate the unique flattened keys
        root_keys = (local_b_idx * current_arena_size) + true_roots
        
        # Allocate fresh indices for unique roots
        unique_root_keys = torch.unique(root_keys)
        num_new_roots = unique_root_keys.numel()
        
        new_ids = torch.arange(current_arena_size, current_arena_size + num_new_roots, device=self.device)
        old_to_new[unique_root_keys] = new_ids
        next_alloc_idx = current_arena_size + num_new_roots
        
        # BFS just like in occurs_check and unify
        frontier_keys = unique_root_keys
        out_nodes, out_is_var, out_children = [], [], []
        
        while frontier_keys.numel() > 0:

            # Decode the flat keys back into local 2D coordinates
            current_local_b_idx = torch.div(frontier_keys, current_arena_size, rounding_mode='floor')
            current_nodes = frontier_keys % current_arena_size
            
            # Fetch structural data for the current batch of nodes
            level_nodes = self.parser.nodes[current_nodes]
            level_is_var = self.parser.is_var_mask[current_nodes]
            level_children = self.parser.children[current_nodes] 
            
            valid_mask = level_children != -1
            
            # expand children
            if valid_mask.any():
                expanded_local_b_idx = current_local_b_idx.unsqueeze(1).expand(-1, self.parser.max_arity)
                
                valid_local_b_idx = expanded_local_b_idx[valid_mask]
                valid_children_old = level_children[valid_mask]
                
                # --- THE SECOND FIX ---
                # Retrieve the original batch ID using the unique_batches array, NOT original_b_idx
                valid_original_b_idx = unique_batches[valid_local_b_idx]
                
                true_children = unifier.update_ref(valid_children_old, valid_original_b_idx)
                
                child_keys = (valid_local_b_idx * current_arena_size) + true_children
                
                # Filter unallocated nodes
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
            self.parser.nodes = torch.cat([self.parser.nodes] + out_nodes)
            self.parser.is_var_mask = torch.cat([self.parser.is_var_mask] + out_is_var)
            self.parser.children = torch.cat([self.parser.children] + out_children)
            
        # Return new roots
        return old_to_new[root_keys]
    
    def decode_term(self, idx, unifier, id_to_symbol, var_name_map, visited=None):
        if visited is None:
            visited = set()
            
        idx_int = int(idx) if isinstance(idx, torch.Tensor) else int(idx)
        
        
        node_tensor = torch.tensor([idx_int], dtype=torch.long, device=self.device)
        b_idx_tensor = torch.tensor([0], dtype=torch.long, device=self.device)
        
        true_idx = unifier.update_ref(node_tensor, b_idx_tensor).item()
        
        
        if true_idx in visited:
            return "[CYCLE DETECTED]"
        visited.add(true_idx)
        
        if unifier.is_var_mask[true_idx]:
            # 1. Try to find the original human-assigned name (e.g., 'X', 'Y')
            idx_to_name = {int(v): k for k, v in var_name_map.items()}
            if true_idx in idx_to_name:
                return idx_to_name[true_idx]
                
            
            var_letters = ['X', 'Y', 'Z', 'V', 'W', 'U']
            assigned_letter = var_letters[true_idx % len(var_letters)]
            
            return f"{assigned_letter}_{true_idx}"
            
        sym_id = unifier.nodes[true_idx].item()
        sym_str = id_to_symbol.get(sym_id, "?")
        
        children = unifier.children[true_idx]
        valid_children = children[children != -1]
        
        if len(valid_children) == 0:
            return sym_str
        else:
            args = [self.decode_term(c.item(), unifier, id_to_symbol, var_name_map, visited.copy()) for c in valid_children]
            return f"{sym_str}({', '.join(args)})"
        
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
                        bound_string = self.decode_term(var_memory_idx, unifier, id_to_symbol, combined_var_map)
                        # Hide trivial self-bindings (e.g., Y = Y)
                        if bound_string != var_name:
                            print(f"      {var_name} = {bound_string}")
            else:
                print("   -> Rejected due to Symbol Mismatch or Occurs Check.")