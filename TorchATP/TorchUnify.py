import torch
import math
from TorchParse import LogicParser

class BatchedGPUUnifier:
    def __init__(self, nodes, children, is_var_mask, max_arity=2):
        self.nodes = nodes
        self.children = children
        self.is_var_mask = is_var_mask
        self.num_nodes = nodes.shape[0]
        self.max_arity = max_arity
        
        # We NO LONGER initialize self.subs here, because we do not know 
        # the batch size yet. It gets allocated freshly inside unify().
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
            # Query the specific row for this pair's batch index
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
            # Flatten frontier and b_idx to feed into the 1D update_ref logic
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
        
        # 1. Initialize the 2D substitution matrix
        base_subs = torch.arange(self.num_nodes, dtype=torch.long, device=self.nodes.device)
        self.subs = base_subs.unsqueeze(0).expand(num_pairs, -1).clone()
        
        frontier = pair_batch
        
        while frontier.shape[0] > 0:
            alive_mask = success_mask[batch_idx]
            frontier = frontier[alive_mask]
            batch_idx = batch_idx[alive_mask]
            
            if frontier.shape[0] == 0: break
            
            # Thread b_idx down into update_ref
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
            
            # --- Bind Variables on LEFT ---
            if torch.any(l_is_v):
                v_idx, t_idx = left[l_is_v], right[l_is_v]
                b_idx = batch_idx[l_is_v]
                
                failed_occurs = self.occurs_check(v_idx, t_idx, b_idx)

                if torch.any(failed_occurs):
                    success_mask[b_idx[failed_occurs]] = False
                
                survivors = ~failed_occurs
                v_surv, t_surv = v_idx[survivors], t_idx[survivors]
                b_surv = b_idx[survivors]
                
                # 2D ASSIGNMENT
                self.subs[b_surv, v_surv] = t_surv
                
                # 2D CONFLICT CHECK
                won_targets = self.subs[b_surv, v_surv]
                conflict_mask = (won_targets != t_surv)
                
                if torch.any(conflict_mask):
                    c_left = won_targets[conflict_mask]
                    c_right = t_surv[conflict_mask]
                    c_batch = b_surv[conflict_mask]
                    next_frontier_pieces.append(torch.stack([c_left, c_right], dim=1))
                    next_batch_pieces.append(c_batch)
                
            # --- Bind Variables on RIGHT ---
            r_bind_mask = r_is_v & ~l_is_v
        
            if torch.any(r_bind_mask):
                v_idx, t_idx = right[r_bind_mask], left[r_bind_mask]
                b_idx = batch_idx[r_bind_mask]
                
                failed_occurs = self.occurs_check(v_idx, t_idx, b_idx)
                
                if torch.any(failed_occurs):
                    success_mask[b_idx[failed_occurs]] = False
                    
                survivors = ~failed_occurs
                v_surv, t_surv = v_idx[survivors], t_idx[survivors]
                b_surv = b_idx[survivors]
                
                # 2D ASSIGNMENT
                self.subs[b_surv, v_surv] = t_surv
                
                # 2D CONFLICT CHECK
                won_targets = self.subs[b_surv, v_surv]
                conflict_mask = (won_targets != t_surv)
                
                if torch.any(conflict_mask):
                    c_left = won_targets[conflict_mask]
                    c_right = t_surv[conflict_mask]
                    c_batch = b_surv[conflict_mask]
                    next_frontier_pieces.append(torch.stack([c_left, c_right], dim=1))
                    next_batch_pieces.append(c_batch)
                
            # --- Handle Function/Constant Symbols ---
            fun_mask = ~l_is_v & ~r_is_v
            f_left, f_right = left[fun_mask], right[fun_mask]
            f_batch = batch_idx[fun_mask]
            
            if f_left.shape[0] > 0:
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
                    
                    next_frontier_pieces.append(torch.stack([c_left[valid_pad], c_right[valid_pad]], dim=1))
                    next_batch_pieces.append(c_batch[valid_pad])
            
            if len(next_frontier_pieces) > 0:
                frontier = torch.cat(next_frontier_pieces, dim=0)
                batch_idx = torch.cat(next_batch_pieces, dim=0)
            else:
                frontier = torch.empty((0, 2), dtype=torch.long, device=self.nodes.device)

        return self.subs, success_mask
    
class SingleUnifierWrapper:
    def __init__(self, subs_row):
        self.subs = subs_row
        
    def update_ref(self, indices, b_idx=None): # Safe default
        curr = indices.clone()
        
    def update_ref(self, indices):
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

class NeuralProverPipeline:
    """
    This class includes both pre- and pos-processing for our batches of claues. 

    1) standardize apart: For now, this can be safely ignored. I added it because in the future we might want to give clauses
    fresh variables when we run Robinson's saturation algorithm. The default you should use is standrize_apart = False. Indeed this 
    is what Stephan Schulz does in PyRes and only deals with creating fresh variables in the resolution algorithm. 

    2) decode term : The raw subsitition tensor for X -> f(Y) would look like [1, 1, 2] in particular it just tells us that X goes to f
    but really X goes to the full term: we need to combine this with the knoweldge of the 
    children tensor. Decode term takes care of this and from [1, 1, 2] + Children reconstructs the subsitution in natural language. 

     
    """
    def __init__(self, max_arity=2, device='cpu'):
        self.parser = LogicParser(max_arity=max_arity)
        self.device = device

    def prove_batch(self, string_pairs, standardize_apart=False):
        """
        Executes batched unification.
        If standardize_apart is False, X on the left is the exact same variable as X on the right.
        If True, X on the left and X on the right are assigned distinct memory indices.
        """
        self.last_run_standardized = standardize_apart
        self.batch_var_map_offset = len(self.parser.var_maps)
        
        if not standardize_apart:
           
            nodes, children, is_var, roots = self.parser.parse_pairs(string_pairs)
            
            nodes = nodes.to(self.device)
            children = children.to(self.device)
            is_var = is_var.to(self.device)
            roots = roots.to(self.device)
            
           
            pair_batch = roots 
            
        else:
   
            flat_strings = []
            for left, right in string_pairs:
                flat_strings.extend([left, right])
                
            nodes, children, is_var, roots = self.parser.parse_clauses(flat_strings)
            
            nodes = nodes.to(self.device)
            children = children.to(self.device)
            is_var = is_var.to(self.device)
            roots = roots.to(self.device)
            
            # Reshape flat roots into [Batch_Size, 2]
            pair_batch = roots.view(-1, 2)
            
        
        unifier = BatchedGPUUnifier(nodes, children, is_var, max_arity=self.parser.max_arity)
        subs, success_mask = unifier.unify(pair_batch)
        
        return subs, success_mask, unifier
    
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
        
        # We instantiate the unifier using the parser's GLOBAL memory arena!
        unifier = BatchedGPUUnifier(
            self.parser.nodes, 
            self.parser.children, 
            self.parser.is_var_mask, 
            max_arity=self.parser.max_arity
        )
        
        subs, success_mask = unifier.unify(pair_batch)
        
        return subs, success_mask, unifier
    

    def instantiate_in_arena(self, root_indices, exclude_idx, unifier):
        valid_roots = [r for i, r in enumerate(root_indices) if i != exclude_idx]
        if not valid_roots:
            return []

        # 1. Evaluate substituted roots
        frontier_old = torch.tensor(valid_roots, dtype=torch.long, device=self.device)
        true_roots = unifier.update_ref(frontier_old)
        
        current_arena_size = self.parser.nodes.shape[0]
        
        # 2. GLOBAL MEMOIZATION TABLE 
        # Maps old true_indices -> newly allocated arena indices
        old_to_new = torch.full((current_arena_size,), -1, dtype=torch.long, device=self.device)
        
        # Extract unique roots to allocate (prevents duplicating identical literals)
        unique_roots = torch.unique(true_roots)
        num_new = unique_roots.numel()
        
        new_ids = torch.arange(current_arena_size, current_arena_size + num_new, device=self.device)
        old_to_new[unique_roots] = new_ids
        next_alloc_idx = current_arena_size + num_new
        
        frontier_unique = unique_roots
        out_nodes, out_is_var, out_children = [], [], []
        
        # 3. BFS Traversal with Deduplication
        while frontier_unique.numel() > 0:
            level_nodes = self.parser.nodes[frontier_unique]
            level_is_var = self.parser.is_var_mask[frontier_unique]
            level_children = self.parser.children[frontier_unique]
            
            valid_mask = level_children != -1
            valid_children_old = level_children[valid_mask]
            
            # Apply substitutions to the children
            true_children_old = unifier.update_ref(valid_children_old)
            
            # FILTER: Which children have NOT been allocated yet?
            unallocated_mask = old_to_new[true_children_old] == -1
            unallocated_children = true_children_old[unallocated_mask]
            
            unique_new_children = torch.unique(unallocated_children)
            num_new_children = unique_new_children.numel()
            
            if num_new_children > 0:
                new_child_ids = torch.arange(next_alloc_idx, next_alloc_idx + num_new_children, device=self.device)
                old_to_new[unique_new_children] = new_child_ids
                next_alloc_idx += num_new_children
                
            # Wire pointers securely using the memoization map
            new_level_children = torch.full_like(level_children, -1)
            new_level_children[valid_mask] = old_to_new[true_children_old]
            
            out_nodes.append(level_nodes)
            out_is_var.append(level_is_var)
            out_children.append(new_level_children)
            
            frontier_unique = unique_new_children
            
        # 4. Push to Global Arena
        if out_nodes:
            self.parser.nodes = torch.cat([self.parser.nodes] + out_nodes)
            self.parser.is_var_mask = torch.cat([self.parser.is_var_mask] + out_is_var)
            self.parser.children = torch.cat([self.parser.children] + out_children)
            
        # Return the exact new roots using the inverse map
        return old_to_new[true_roots].tolist()
    
    def decode_term(self, idx, unifier, id_to_symbol, var_name_map, visited=None):
        if visited is None:
            visited = set()
            
        idx_int = int(idx) if isinstance(idx, torch.Tensor) else int(idx)
        
        # --- THE FIX: Provide a dummy batch index (0) for decoding ---
        node_tensor = torch.tensor([idx_int], dtype=torch.long, device=self.device)
        b_idx_tensor = torch.tensor([0], dtype=torch.long, device=self.device)
        
        true_idx = unifier.update_ref(node_tensor, b_idx_tensor).item()
        # -------------------------------------------------------------
        
        if true_idx in visited:
            return "[CYCLE DETECTED]"
        visited.add(true_idx)
        
        if unifier.is_var_mask[true_idx]:
            # 1. Try to find the original human-assigned name (e.g., 'X', 'Y')
            idx_to_name = {int(v): k for k, v in var_name_map.items()}
            if true_idx in idx_to_name:
                return idx_to_name[true_idx]
                
            # 2. Fallback: Generate a fresh, readable variable name based on the index
            # Cycle through a pool of standard variable characters
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