import torch
import re
import TorchUnify
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
    
    dummy_unifier = TorchUnify.BatchedGPUUnifier(
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
    Parses a TPTP string and loads its terms directly into the TorchParse memory arena (
    this is a side effect!) Returns a VirtualClause containing tuples of (is_negative_bool, root_index_int).
    Safely toggles between List and Tensor states to prevent GPU pipeline crashes.
    """
    # Extract the name and the literal block
    match = re.match(r"cnf\(([^,]+),\s*[^,]+,\s*\((.*)\)\s*\)\.?", tptp_str.strip())
    if not match:
        raise ValueError(f"Could not parse TPTP string: {tptp_str}")
        
    name = match.group(1).strip()
    literal_block = match.group(2).strip()
    
    #  Split by the OR operator
    raw_literals = literal_block.split('|')
    
    # A single clause must share one variable map for factoring to work
    local_vars = {}
    virtual_literals = []
    
    # State-Saver for PyTorch Tensors
    was_tensor = False
    device = 'cpu'
    if isinstance(pipeline.parser.nodes, torch.Tensor):
        was_tensor = True
        device = pipeline.parser.nodes.device
        pipeline.parser.nodes = pipeline.parser.nodes.tolist()
        pipeline.parser.children = pipeline.parser.children.tolist()
        pipeline.parser.is_var_mask = pipeline.parser.is_var_mask.tolist()

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
        
    # Restore the arena : If it was a GPU tensor, push the new nodes to VRAM ---
    if was_tensor:
        pipeline.parser.nodes = torch.tensor(pipeline.parser.nodes, dtype=torch.long, device=device)
        pipeline.parser.children = torch.tensor(pipeline.parser.children, dtype=torch.long, device=device)
        pipeline.parser.is_var_mask = torch.tensor(pipeline.parser.is_var_mask, dtype=torch.bool, device=device)
        
    # Save the clause's variable map into the pipeline's history for debugging
    pipeline.parser.var_maps.append(local_vars)
        
    return VirtualClause(virtual_literals, name=name)