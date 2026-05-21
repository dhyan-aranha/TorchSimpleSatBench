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

        # Initialize substitution tensor: every node points to itself initially
        self.subs = torch.arange(self.num_nodes, dtype=torch.long, device=nodes.device)

    def update_ref(self, indices):
        """
        Morally the same as update_ref in TorchUnifyNaive
        """
        curr = indices.clone()
        valid_mask = (curr != -1)
        active = valid_mask.clone()
        
        # Only loop while at least one pointer is still changing
        while active.any():
            nxt = curr.clone()
            # Only query self.subs for currently active, valid nodes
            nxt[active] = self.subs[curr[active]]
            
            # Determine which nodes changed on this hop
            changed = (nxt != curr) & active
            curr = nxt
            active = changed
            
        return curr
    
    def occurs_check(self, var_indices, target_indices):
        """
        Dynamically bounds the Breadth-First Search.
        No hardcoded limits. Terminates naturally when the deepest leaf is reached.
        """
        K = var_indices.shape[0]
        if K == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.subs.device)

        device = self.subs.device
        failed_occurs = torch.zeros(K, dtype=torch.bool, device=device)
        
        # Initial shape: [K, 1]
        frontier = target_indices.unsqueeze(1) 
        
        # LOOP CONDITION: Continue as long as there is at least one active node 
        # anywhere in the entire batch's frontier.
        while frontier.shape[1] > 0:
            
            frontier = self.update_ref(frontier)
            
            matches = (frontier == var_indices.unsqueeze(1))
            new_fails = matches.any(dim=1)
            failed_occurs = failed_occurs | new_fails
            
            # EARLY STOP 1: If every target in the batch has failed, 
            # we can stop expanding immediately.
            active_rows = ~failed_occurs
            if not active_rows.any():
                break
                
            valid_mask = (frontier != -1)
            safe_frontier = frontier.clone()
            safe_frontier[~valid_mask] = 0 
            
            next_frontier = self.children[safe_frontier] 
            next_frontier[~valid_mask] = -1
            
            frontier = next_frontier.view(K, -1)
            
            # EARLY STOP 2 (The Guarantor of Termination): 
            # Drop columns that are entirely -1 across the batch.
            col_has_data = (frontier != -1).any(dim=0)
            frontier = frontier[:, col_has_data]

        return failed_occurs

    #@torch.compile(dynamic=True, mode="reduce-overhead")
    def unify(self, pair_batch):
        """
        Iterative, batched unification loop.
        """
        num_pairs = pair_batch.shape[0]

        # Tracks which clauses in the batch are still viable
        success_mask = torch.ones(num_pairs, dtype=torch.bool, device=self.nodes.device)
        batch_idx = torch.arange(num_pairs, dtype=torch.long, device=self.nodes.device)
        
        # Reset substitutions for the new batch
        self.subs = torch.arange(self.num_nodes, dtype=torch.long, device=self.nodes.device)
        frontier = pair_batch
        
        while frontier.shape[0] > 0:
            # Prune dead branches
            alive_mask = success_mask[batch_idx]
            frontier = frontier[alive_mask]
            batch_idx = batch_idx[alive_mask]
            
            if frontier.shape[0] == 0: break
            
            # Update left and right pointers
            left = self.update_ref(frontier[:, 0])
            right = self.update_ref(frontier[:, 1])
            
            # Filter trivial matches (e.g., X = X)
            active = (left != right)
            left, right = left[active], right[active]
            batch_idx = batch_idx[active] 
            
            if left.shape[0] == 0: break
            
            l_is_v = self.is_var_mask[left]
            r_is_v = self.is_var_mask[right]
            
            next_frontier_pieces = []
            next_batch_pieces = []
            
            # Binding variables on the LEFT hand side
            if torch.any(l_is_v):
                v_idx, t_idx = left[l_is_v], right[l_is_v]
                b_idx = batch_idx[l_is_v]
                
                failed_occurs = self.occurs_check(v_idx, t_idx)

                if torch.any(failed_occurs):
                    success_mask[b_idx[failed_occurs]] = False
                
                survivors = ~failed_occurs
                v_surv, t_surv = v_idx[survivors], t_idx[survivors]
                
                self.subs[v_surv] = t_surv
                
                # Intra-clause conflict check (e.g., f(X, X) = f(a, b))
                won_targets = self.subs[v_surv]
                conflict_mask = (won_targets != t_surv)
                
                if torch.any(conflict_mask):
                    c_left = won_targets[conflict_mask]
                    c_right = t_surv[conflict_mask]
                    c_batch = b_idx[survivors][conflict_mask]
                    next_frontier_pieces.append(torch.stack([c_left, c_right], dim=1))
                    next_batch_pieces.append(c_batch)
                
            # Binding variables on the RIGHT hand side 
            r_bind_mask = r_is_v & ~l_is_v
        
            if torch.any(r_bind_mask):
                v_idx, t_idx = right[r_bind_mask], left[r_bind_mask]
                b_idx = batch_idx[r_bind_mask]
                
                failed_occurs = self.occurs_check(v_idx, t_idx)
                
                if torch.any(failed_occurs):
                    success_mask[b_idx[failed_occurs]] = False
                    
                survivors = ~failed_occurs
                v_surv, t_surv = v_idx[survivors], t_idx[survivors]
                
                self.subs[v_surv] = t_surv
                
                won_targets = self.subs[v_surv]
                conflict_mask = (won_targets != t_surv)
                
                if torch.any(conflict_mask):
                    c_left = won_targets[conflict_mask]
                    c_right = t_surv[conflict_mask]
                    c_batch = b_idx[survivors][conflict_mask]
                    next_frontier_pieces.append(torch.stack([c_left, c_right], dim=1))
                    next_batch_pieces.append(c_batch)
                
            #  Handling Function/Constant Symbols
            fun_mask = ~l_is_v & ~r_is_v
            f_left, f_right = left[fun_mask], right[fun_mask]
            f_batch = batch_idx[fun_mask]
            
            if f_left.shape[0] > 0:
                # Check for symbol mismatch (e.g., f(...) != g(...))
                mismatch = (self.nodes[f_left] != self.nodes[f_right])
                if torch.any(mismatch):
                    success_mask[f_batch[mismatch]] = False
                
                valid_struct = ~mismatch
                f_left, f_right = f_left[valid_struct], f_right[valid_struct]
                f_batch = f_batch[valid_struct]
                
                if f_left.shape[0] > 0:
                    c_left = self.children[f_left]
                    c_right = self.children[f_right]
                    
                    # Duplicate parent batch ID for each child
                    c_batch = f_batch.unsqueeze(1).expand(-1, self.max_arity).flatten()
                    c_left, c_right = c_left.flatten(), c_right.flatten()
                    
                    valid_pad = (c_left != -1) & (c_right != -1)
                    
                    next_frontier_pieces.append(torch.stack([c_left[valid_pad], c_right[valid_pad]], dim=1))
                    next_batch_pieces.append(c_batch[valid_pad])
            
            # Rebuild the frontier for the next loop
            if len(next_frontier_pieces) > 0:
                frontier = torch.cat(next_frontier_pieces, dim=0)
                batch_idx = torch.cat(next_batch_pieces, dim=0)
            else:
                frontier = torch.empty((0, 2), dtype=torch.long, device=self.nodes.device)

        return self.subs, success_mask

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
        """
        Deep copies the selected literals into the global memory arena,
        applying the substitutions found during unification.
        Returns a list of the new integer root indices.
        """
        new_roots = []
        
        # Buffers to hold the new node data before we bulk-append it to the global arena
        new_nodes_buffer = []
        new_is_var_buffer = []
        new_children_buffer = []
        
        # We need the current size of the global arena so our new pointers align correctly
        current_arena_size = self.parser.nodes.shape[0]
        
        def _copy_tree(node_idx):
            # 1. Ask the unifier what this node is bound to
            node_tensor = torch.tensor([node_idx], dtype=torch.long, device=self.device)
            true_idx = unifier.update_ref(node_tensor).item()
            
            # 2. Process children first (Post-order traversal)
            children = self.parser.children[true_idx]
            new_child_indices = []
            
            for c in children:
                if c.item() != -1:
                    new_child_indices.append(_copy_tree(c.item()))
                else:
                    new_child_indices.append(-1)
                    
            # 3. Now process self
            sym_id = self.parser.nodes[true_idx].item()
            is_v = self.parser.is_var_mask[true_idx].item()
            
            # Calculate what index this new node will have once appended to the global arena
            my_new_idx = current_arena_size + len(new_nodes_buffer)
            
            new_nodes_buffer.append(sym_id)
            new_is_var_buffer.append(is_v)
            
            # Pad the children array to match max_arity
            padded_children = new_child_indices + [-1] * (self.parser.max_arity - len(new_child_indices))
            new_children_buffer.append(padded_children)
            
            return my_new_idx

        # --- Execution ---
        # Generate the copies for every literal except the one that was resolved away
        for i, root in enumerate(root_indices):
            if i == exclude_idx:
                continue
            new_roots.append(_copy_tree(root))
            
        # Bulk append the new nodes to the global arena using PyTorch Cat
        if new_nodes_buffer:
            self.parser.nodes = torch.cat([
                self.parser.nodes, 
                torch.tensor(new_nodes_buffer, dtype=torch.long, device=self.device)
            ])
            self.parser.is_var_mask = torch.cat([
                self.parser.is_var_mask, 
                torch.tensor(new_is_var_buffer, dtype=torch.bool, device=self.device)
            ])
            self.parser.children = torch.cat([
                self.parser.children, 
                torch.tensor(new_children_buffer, dtype=torch.long, device=self.device)
            ])
            
        return new_roots
    
    def decode_term(self, idx, unifier, id_to_symbol, var_name_map, visited=None):
        if visited is None:
            visited = set()
            
        idx_int = int(idx) if isinstance(idx, torch.Tensor) else int(idx)
        true_idx = unifier.update_ref(torch.tensor([idx_int], device=self.device)).item()
        
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