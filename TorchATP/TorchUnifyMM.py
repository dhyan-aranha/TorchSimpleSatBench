import torch
from TorchParse import LogicParser

def batched_unify_mm_packed(nodes, children, is_var_mask, roots):
    """
    Martelli-Montanari unification operating directly on the packed memory arena.
    Produces a [B, Total_N] substitution tensor without padding or pointer shifting.
    """
    Total_N = nodes.size(0)
    B = roots.size(0)
    max_arity = children.size(1)
    device = nodes.device
    
    # 1. Arithmetic Validity Mask [B, Total_N]
    # Replaces physical padding by mathematically walling off out-of-bounds nodes per batch.
    starts = roots[:, 0]
    ends = torch.cat([starts[1:], torch.tensor([Total_N], device=device)])
    
    idx_matrix = torch.arange(Total_N, device=device).unsqueeze(0).expand(B, Total_N)
    valid_mask = (idx_matrix >= starts.unsqueeze(1)) & (idx_matrix < ends.unsqueeze(1))
    
    # 2. State Tensors [B, Total_N]
    parents = idx_matrix.clone()
    status = torch.zeros(B, dtype=torch.long, device=device) # 0: Active, 1: Success, -1: Clash, -2: Cycle
    processed = torch.zeros((B, Total_N), dtype=torch.bool, device=device)
    
    def batched_find(p, indices):
        curr = indices.clone()
        for _ in range(Total_N): # Bounded by absolute maximum graph depth
            nxt_vals = torch.gather(p, 1, curr)
            if torch.equal(curr, nxt_vals):
                break
            curr = nxt_vals
        return curr

    # 3. Global Initial In-degrees (Counters)
    # Because the parsed graph is fixed, base counters can be computed once globally
    valid_c = children[children != -1]
    base_counters = torch.bincount(valid_c, minlength=Total_N)
    counters = base_counters.unsqueeze(0).expand(B, Total_N).clone()
    
    # 4. Initial Equation Setup
    r1 = roots[:, 0].unsqueeze(1)
    r2 = roots[:, 1].unsqueeze(1)
    parents.scatter_(1, r2, r1)
    
    # 5. Main Execution Loop
    while True:
        active_mask = (status == 0)
        if not active_mask.any():
            break 
            
        parents = batched_find(parents, idx_matrix)
        
        # --- Topological Selection ---
        is_root = (parents == idx_matrix)
        eligible = is_root & (~processed) & (counters == 0) & valid_mask
        has_eligible = eligible.any(dim=1)
        
        # A batch is only "done" when all its ROOTS are processed.
        # Non-root nodes belong to a processed class, so they are safely ignored.
        unprocessed_roots = is_root & (~processed) & valid_mask
        done = ~unprocessed_roots.any(dim=1)
        
        status[(~has_eligible) & done & active_mask] = 1
        status[(~has_eligible) & (~done) & active_mask] = -2
        
        active_mask = (status == 0)
        if not active_mask.any():
            break
            
        selected_roots = eligible.int().argmax(dim=1)
        
        # --- Clash Check ---
        f_nodes_mask = (parents == selected_roots.unsqueeze(1)) & (~is_var_mask) & valid_mask
        has_functions = f_nodes_mask.any(dim=1)
        
        # Broadcast 1D global nodes to 2D for masking
        nodes_2d = nodes.unsqueeze(0).expand(B, Total_N)
        max_sym = torch.where(f_nodes_mask, nodes_2d, torch.tensor(-1, device=device)).max(dim=1)[0]
        min_sym = torch.where(f_nodes_mask, nodes_2d, torch.tensor(torch.iinfo(torch.long).max, device=device)).min(dim=1)[0]
        
        clash_mask = (max_sym != min_sym) & has_functions & active_mask
        status[clash_mask] = -1
        active_mask = (status == 0) 
        
        # --- Term Reduction & Compactification ---
        for j in range(max_arity):
            # Expand the global 1D child column to [B, Total_N]
            child_col = children[:, j].unsqueeze(0).expand(B, Total_N)
            
            active_children = torch.where(f_nodes_mask, child_col, torch.tensor(-1, device=device))
            has_child = (active_children != -1)
            
            # Safely get roots for all active children (using dummy 0 for -1 padding)
            safe_active = torch.maximum(active_children, torch.tensor(0, dtype=torch.long, device=device))
            all_child_roots = batched_find(parents, safe_active)
            
            decrement_mask = has_child & active_mask.unsqueeze(1)
            
            if decrement_mask.any():
                rel_b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, Total_N)[decrement_mask]
                rel_c_roots = all_child_roots[decrement_mask]
                
                counters.index_put_((rel_b_idx, rel_c_roots), torch.tensor(-1, device=device), accumulate=True)
            
            # Elect a leader for this column and merge the frontier
            first_child_col = has_child.int().argmax(dim=1)
            leader_children = active_children[torch.arange(B, device=device), first_child_col]
            
            # SAFEGUARD: Prevent torch.gather crash on -1 padding
            safe_leader_children = torch.maximum(leader_children, torch.tensor(0, dtype=torch.long, device=device))
            leader_roots = batched_find(parents, safe_leader_children.unsqueeze(1)).squeeze(1)
            
            if decrement_mask.any():
                c_roots = all_child_roots[decrement_mask]
                l_roots = leader_roots[rel_b_idx]
                
                # SAFEGUARD: Only union classes that are not already identical!
                # If they are already the same, adding/zeroing destroys the dependency tally.
                union_mask = (c_roots != l_roots)
                
                if union_mask.any():
                    u_b_idx = rel_b_idx[union_mask]
                    u_c_roots = c_roots[union_mask]
                    u_l_roots = l_roots[union_mask]
                    
                    # Link classes 
                    parents.index_put_((u_b_idx, u_c_roots), u_l_roots)
                    
                    # Merge counters
                    c_counters = counters[u_b_idx, u_c_roots]
                    counters.index_put_((u_b_idx, u_l_roots), c_counters, accumulate=True)
                    counters.index_put_((u_b_idx, u_c_roots), torch.zeros_like(c_counters))
        
        processed[torch.arange(B, device=device), selected_roots] = True

    parents = batched_find(parents, idx_matrix)
    
    class_reps = idx_matrix.clone()
    is_function = (~is_var_mask) & valid_mask
    
    if is_function.any():
        b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, Total_N)[is_function]
        f_roots = parents[is_function]
        f_indices = idx_matrix[is_function]
        
        # Override the variable DSU root with the physical index of the function frame
        class_reps.index_put_((b_idx, f_roots), f_indices)
        
    subs = torch.gather(class_reps, 1, parents)
    
    # Nullify bindings outside the valid pair boundary
    subs[~valid_mask] = -1
    
    return status, subs



def decode_term(b, node_idx, subs, nodes, children, is_var_mask, vocab_inv, var_inv, visited=None):
    """Recursively decodes the substitution matrix back into a human-readable string."""
    if visited is None:
        visited = set()
        
    rep_idx = subs[b, node_idx].item()
    
    if rep_idx == -1: return "?"
    if rep_idx in visited: return "[CYCLE]"
    
    visited.add(rep_idx)
    
    if is_var_mask[rep_idx]:
        return var_inv.get(rep_idx, f"V_{rep_idx}")
    else:
        sym_str = vocab_inv.get(nodes[rep_idx].item(), "?")
        c_list = children[rep_idx]
        valid_c = c_list[c_list != -1]
        
        if len(valid_c) == 0:
            return sym_str
        else:
            args = [decode_term(b, c.item(), subs, nodes, children, is_var_mask, vocab_inv, var_inv, visited.copy()) for c in valid_c]
            return f"{sym_str}({', '.join(args)})"


