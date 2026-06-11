# A version of torch unify with a custom CUDA kernels handling, pointer chasing and unification. 
import torch
import math
from TorchParse import LogicParser
from torch.utils.cpp_extension import load

cuda_unify = load(
    name="cuda_unify", 
    sources=["unify_kernel.cu"], 
    verbose=True
)

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
        Maintains standard PyTorch behavior for small queries 
        (used by occurs_check and instantiate).
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
        K = var_indices.shape[0]
        if K == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.nodes.device)

        device = self.nodes.device
        failed_occurs = torch.zeros(K, dtype=torch.bool, device=device)
        
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
        num_pairs = pair_batch.shape[0]

        global_success_mask = torch.ones(num_pairs, dtype=torch.bool, device=self.nodes.device)
        base_subs = torch.arange(self.num_nodes, dtype=torch.long, device=self.nodes.device)
        
        # 1. Cast ALL matrices to int32 for C++ compatibility upfront
        subs_int32 = base_subs.unsqueeze(0).expand(num_pairs, -1).clone().to(torch.int32).contiguous()
        c_nodes = self.nodes.to(torch.int32).contiguous()
        c_children = self.children.to(torch.int32).contiguous()
        
        frontier_left = pair_batch[:, 0].to(torch.int32)
        frontier_right = pair_batch[:, 1].to(torch.int32)
        batch_idx_int32 = torch.arange(num_pairs, dtype=torch.int32, device=self.nodes.device)
        
        while frontier_left.shape[0] > 0:
            alive_mask = global_success_mask[batch_idx_int32]
            frontier_left = frontier_left[alive_mask]
            frontier_right = frontier_right[alive_mask]
            current_batch_idx = batch_idx_int32[alive_mask]
            
            if frontier_left.shape[0] == 0: break
            
            # 2. Fully compress the DAG using update_ref_kernel
            subs_int32 = cuda_unify.launch_update_ref(subs_int32, self.num_nodes)
            
            # Since subs_int32 is now completely flat, finding roots is a direct 2D read
            left_roots = subs_int32[current_batch_idx, frontier_left]
            right_roots = subs_int32[current_batch_idx, frontier_right]
            
            # 3. Execute atomic bindings and discover collisions
            wave_success, next_left, next_right, next_batch = cuda_unify.launch_unify(
                left_roots, 
                right_roots, 
                current_batch_idx, 
                subs_int32,  # Modified in-place via atomicCAS!
                self.is_var_mask, 
                c_nodes, 
                c_children, 
                self.num_nodes, 
                self.max_arity
            )
            
            global_success_mask[current_batch_idx[~wave_success]] = False
            
            frontier_left = next_left
            frontier_right = next_right
            batch_idx_int32 = next_batch

        # 4. Save the fully resolved matrix back as standard PyTorch longs
        self.subs = subs_int32.to(torch.long)
        
        bound_mask = self.subs != base_subs.unsqueeze(0).expand(num_pairs, -1)
        b_idx, v_idx = torch.where(bound_mask)
        t_idx = self.subs[bound_mask]
        
        failed_occurs = self.occurs_check(v_idx, t_idx, b_idx)
        
        if torch.any(failed_occurs):
            global_success_mask[b_idx[failed_occurs]] = False

        return self.subs, global_success_mask


class ProverPipeline:
    def __init__(self, device='cpu', max_arity=0):
        self.device = device
        self.parser = LogicParser()
        if max_arity > 0:
            self.parser.max_arity = max_arity
    
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

        current_arena_size = self.parser.nodes.shape[0]
        
        original_b_idx = batched_requests[:, 0]
        roots_old = batched_requests[:, 1]
        
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
        
        # 2D memoization table
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