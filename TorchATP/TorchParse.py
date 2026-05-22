import torch
from lexer import Lexer, Token 

class LogicParser:
    def __init__(self, max_arity=2):
        self.global_vocab = {}
        self.next_vocab_id = 1
        self.max_arity = max_arity

        self.var_maps = []
        
        self.nodes = []
        self.children = []
        self.is_var_mask = []

    def parse_clauses(self, clause_strings):
        """This will be used in the future when we want to standardize apart."""
        roots = []
        
        for clause in clause_strings:
            # 2. Initialize the Lexer for this specific clause
            lexer = Lexer(clause)
            local_vars = {}
            
            # Pass the lexer state directly into the recursive parser
            root_idx = self._parse_term(lexer, local_vars)
            roots.append(root_idx)
            
            self.var_maps.append(local_vars) 
            
        return (
            torch.tensor(self.nodes),
            torch.tensor(self.children),
            torch.tensor(self.is_var_mask, dtype=torch.bool),
            torch.tensor(roots)
        )
    
    def parse_pairs(self, string_pairs):
        """Parsing for unification without worrying about standardizing apart."""
        roots = []
        
        for left_str, right_str in string_pairs:
            local_vars = {} 
            
            # Parse Left Side
            lex_left = Lexer(left_str)
            left_root = self._parse_term(lex_left, local_vars)
            
            # Parse Right Side 
            lex_right = Lexer(right_str)
            right_root = self._parse_term(lex_right, local_vars)
            
            # Save them as a pair [Left_ID, Right_ID]
            roots.append([left_root, right_root])
            
            self.var_maps.append(local_vars)
            
        return (
            torch.tensor(self.nodes),
            torch.tensor(self.children),
            torch.tensor(self.is_var_mask, dtype=torch.bool),
            torch.tensor(roots) # Shape will be [Batch_Size, 2]
        )
    
    def _parse_term(self, lexer, local_vars):
        VAR_DUMMY_ID = -2
        tok = lexer.Next()
        
        if tok.type == Token.EOFToken:
            raise ValueError("Unexpected end of string")
            
        
        # Reject Quantifiers
        if tok.type in (Token.Universal, Token.Existential):
            raise ValueError(
                f"TorchParse expects quantifier-free clauses (CNF). "
                f"Found quantifier '{tok.literal}'. Ensure preprocessing is run first."
            )
            
        symbol = tok.literal
        is_var = (tok.type == Token.IdentUpper)
        
        
        if is_var:
            if symbol not in local_vars:
                # First time seeing this variable in this clause
                idx = len(self.nodes)
                self.nodes.append(VAR_DUMMY_ID) # Dummy Vocab ID for all variables
                self.is_var_mask.append(True)

                
                self.children.append([-1] * self.max_arity)
                local_vars[symbol] = idx 
            
            # Return existing memory address if seen before
            return local_vars[symbol]
            
        
        else:
            if symbol not in self.global_vocab:
                self.global_vocab[symbol] = self.next_vocab_id
                self.next_vocab_id += 1
                
            idx = len(self.nodes)
            self.nodes.append(self.global_vocab[symbol])
            self.is_var_mask.append(False)
            self.children.append([-1] * self.max_arity) 
            
            # Check if this symbol has arguments (a following '(')
            if lexer.Look().type == Token.OpenPar:
                lexer.AcceptTok(Token.OpenPar) # consume '('
                
                arg_indices = []
                while lexer.Look().type != Token.ClosePar:
                    if lexer.Look().type == Token.Comma:
                        lexer.AcceptTok(Token.Comma) # consume ','
                        continue
                    
                    # Recursively parse the argument
                    arg_idx = self._parse_term(lexer, local_vars)
                    arg_indices.append(arg_idx)
                    
                lexer.AcceptTok(Token.ClosePar) # consume ')'
                
                # Pad with -1 to reach max_arity
                while len(arg_indices) < self.max_arity:
                    arg_indices.append(-1)
                    
                self.children[idx] = arg_indices[:self.max_arity]
                
            return idx

