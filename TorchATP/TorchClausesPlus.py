import torch
import re
import TorchUnifyPlus
from derivations import Derivable, flatDerivation
from lexer import Lexer


class Literal:
    def __init__(self, atom_string, negative=False):
        self.atom_string = atom_string.strip()
        self.negative = negative

    def __repr__(self):
        sign = "~" if self.negative else ""
        return f"{sign}{self.atom_string}"

    def is_negative(self):
        return self.negative

    def instantiate(self, subst_dict):
        """
        Applies a dictionary of string bindings (e.g., {"X": "butler"}) 
        to the literal using regex word boundaries to prevent partial matches.
        """
        new_str = self.atom_string
        for var_name, bound_val in subst_dict.items():
            # \b ensures we replace "X" but not the "X" inside "Xylophone"
            new_str = re.sub(rf'\b{var_name}\b', bound_val, new_str)
            
        return Literal(new_str, self.negative)
    
class Clause(Derivable):
    def __init__(self, literals, name=None):
        self.literals = literals
        super().__init__(name)
    
    def __repr__(self):
        lit_str = " | ".join(map(repr, self.literals))
        return f"cnf({self.name}, plain, ({lit_str}))"
    
    def is_empty(self):
        return len(self.literals) == 0
    
    def remove_duplicates(self):
        unique_lits = []
        seen = set()
        for l in self.literals:
            # remove duplicates
            rep = repr(l)
            if rep not in seen:
                seen.add(rep)
                unique_lits.append(l)
        self.literals = unique_lits

class VirtualClause(Derivable):
    def __init__(self, literals, name=None):
        """
        literals: A list of tuples -> [(is_negative_bool, root_idx_int), ...]
        """
        self.literals = literals 
        Derivable.__init__(self, name)

    def is_empty(self):
        return len(self.literals) == 0
        
    def deduplicate(self):
        self.literals = list(set(self.literals))

def decode_virtual_clause(v_clause, pipeline):
    """
    Helper function to translate a VirtualClause (tuples) back to a string.
    Only call this when strictly necessary (e.g., inside a DEBUG check).
    """
    if v_clause.is_empty():
        return "[] (Empty Clause)"
        
    id_to_symbol = {v: k for k, v in pipeline.parser.global_vocab.items()}
    dummy_var_map = {} 
    
    
    dummy_unifier = TorchUnifyPlus.BatchedGPUUnifier(
        pipeline.parser.nodes, 
        pipeline.parser.children, 
        pipeline.parser.is_var_mask, 
        max_arity=pipeline.parser.max_arity
    )
    
    lit_strings = []
    
    for is_neg, root_idx in v_clause.literals:
        
        lit_str = pipeline.decode_term(root_idx, dummy_unifier, id_to_symbol, dummy_var_map)
        
        if is_neg:
            lit_str = f"~{lit_str}"
            
        lit_strings.append(lit_str)
        
    return " | ".join(lit_strings)

def parse_tptp_string(tptp_str):
    """
    Parses cnf(<name>, <type>, <literal list>)
    """
    match = re.match(r"cnf\(([^,]+),\s*[^,]+,\s*\((.*)\)\s*\)", tptp_str.strip())
    if not match:
        raise ValueError(f"Could not parse TPTP string: {tptp_str}")
    
    name = match.group(1).strip()
    literal_block = match.group(2).strip()

    
    raw_literals = literal_block.split('|')

    
    parsed_literals = []
    for raw_lit in raw_literals:
        raw_lit = raw_lit.strip()
        is_neg = raw_lit.startswith('~')
        atom_str = raw_lit[1:] if is_neg else raw_lit

        parsed_literals.append(Literal(atom_str, negative=is_neg))

    return Clause(parsed_literals, name=name)

def parse_tptp_to_virtual_clause(tptp_str, pipeline):
    """
    Parses a TPTP string and loads its terms directly into the TorchParse memory arena.
    Returns a VirtualClause containing tuples of (is_negative_bool, root_index_int).
    Safely toggles between List and Tensor states to prevent GPU pipeline crashes.
    """
    
    match = re.match(r"cnf\(([^,]+),\s*[^,]+,\s*\((.*)\)\s*\)\.?", tptp_str.strip())
    if not match:
        
        match = re.match(r"cnf\(([^,]+),\s*[^,]+,\s*(.*)\)\.?", tptp_str.strip())
        
    if not match:
        raise ValueError(f"Could not parse TPTP string: {tptp_str}")
        
    name = match.group(1).strip()
    literal_block = match.group(2).strip()
    
    
    raw_literals = literal_block.split('|')
    
    
    local_vars = {}
    virtual_literals = []
    
    
    was_tensor = False
    device = 'cpu'
    if isinstance(pipeline.parser.nodes, torch.Tensor):
        was_tensor = True
        device = pipeline.parser.nodes.device
        ptr = pipeline.parser.arena_ptr
        
        # Save references to tensors so we don't delete them
        saved_nodes_tensor = pipeline.parser.nodes
        saved_children_tensor = pipeline.parser.children
        saved_mask_tensor = pipeline.parser.is_var_mask
        
        # Extract only the active data slice and convert to fast Python lists
        pipeline.parser.nodes = saved_nodes_tensor[:ptr].tolist()
        pipeline.parser.children = saved_children_tensor[:ptr].tolist()
        pipeline.parser.is_var_mask = saved_mask_tensor[:ptr].tolist()
    

    # Parse each literal into the continuous global arena
    for raw_lit in raw_literals:
        raw_lit = raw_lit.strip()
        
        # Extract and strip the polarity
        is_neg = raw_lit.startswith('~')
        atom_str = raw_lit[1:] if is_neg else raw_lit
        atom_str = atom_str.strip()
        
        # Parse the pure positive atom into the memory arena
        lexer = Lexer(atom_str)
        root_idx = pipeline.parser._parse_term(lexer, local_vars)
        
        # Save as a tuple
        virtual_literals.append((is_neg, root_idx))
        
    
    if was_tensor:
        new_len = len(pipeline.parser.nodes)
        num_new = new_len - ptr
        
        if num_new > 0:
            if ptr + num_new > saved_nodes_tensor.shape[0]:
                raise MemoryError("VRAM Arena Capacity Exceeded during parsing!")
                
            # 1. Extract only the newly appended items from the lists
            new_nodes = torch.tensor(pipeline.parser.nodes[ptr:], dtype=torch.long, device=device)
            new_children = torch.tensor(pipeline.parser.children[ptr:], dtype=torch.long, device=device)
            new_mask = torch.tensor(pipeline.parser.is_var_mask[ptr:], dtype=torch.bool, device=device)
            
            # 2. Write them directly into the empty space of VRAM buffer
            saved_nodes_tensor[ptr : ptr + num_new] = new_nodes
            saved_children_tensor[ptr : ptr + num_new] = new_children
            saved_mask_tensor[ptr : ptr + num_new] = new_mask
            
            # 3. Update the global pointer
            pipeline.parser.arena_ptr += num_new
            
        # 4. Restore the pipeline to use the massive tensors again
        pipeline.parser.nodes = saved_nodes_tensor
        pipeline.parser.children = saved_children_tensor
        pipeline.parser.is_var_mask = saved_mask_tensor
    
        
    # Save the clause's variable map into the pipeline's history for debugging
    pipeline.parser.var_maps.append(local_vars)
        
    return VirtualClause(virtual_literals, name=name)