import torch
#import TorchUnify
import TorchUnifyPlus
import TorchParse
import TorchSat
import TorchSimpBatchPlus
from TorchClauses import parse_tptp_to_virtual_clause
import TorchClausesPlus
from derivations import  enableDerivationOutput

easier_tptp_strings = [
    "cnf(socrates_is_man, axiom, (man(socrates))).",
    "cnf(all_men_mortal, axiom, (~man(X) | mortal(X))).",
    "cnf(prove_socrates_mortal, negated_conjecture, (~mortal(socrates)))."
]

tptp_string_trace_test = [
    "cnf(ax1, axiom, (p(X) | ~p(f(X)))).",
    "cnf(ax2, axiom, (p(a)))."
]

tptp_res_test = [
    "cnf(c1,axiom,(p(a, X)|p(X,a))).",
    "cnf(c2,axiom,(~p(a,b)|p(f(Y),a))).",
    "cnf(c3,axiom,(p(Z,X)|~p(f(Z),X0))).",
    "cnf(c4,axiom,p(X,X)|p(a,f(Y))).",  
    "cnf(c5,axiom,p(X)|~q|p(a)|~q|p(Y)).",
    "cnf(not_p,axiom, (~p(a))).",
    "cnf(taut,axiom,(p(X4)|~p(X4)))."
]

pipeline = TorchUnifyPlus.NeuralProverPipeline(device='cpu')

given_clause = TorchClausesPlus.parse_tptp_to_virtual_clause(tptp_res_test[1], pipeline)
processed_clause = [TorchClausesPlus.parse_tptp_to_virtual_clause( tptp_res_test[2], pipeline)]


# --- THE PRE-ALLOCATED TENSOR TRANSITION ---
CAPACITY = 5_000_000  # VRAM Buffer Size
initial_size = len(pipeline.parser.nodes)

# Initialize the critical tracking pointer
pipeline.parser.arena_ptr = initial_size 

# Allocate massive empty tensors
new_nodes = torch.zeros(CAPACITY, dtype=torch.long, device=pipeline.device)
new_children = torch.full((CAPACITY, pipeline.parser.max_arity), -1, dtype=torch.long, device=pipeline.device)
new_is_var = torch.zeros(CAPACITY, dtype=torch.bool, device=pipeline.device)

# Copy the starting axioms into the top of the buffer
new_nodes[:initial_size] = torch.tensor(pipeline.parser.nodes, dtype=torch.long, device=pipeline.device)
new_children[:initial_size] = torch.tensor(pipeline.parser.children, dtype=torch.long, device=pipeline.device)
new_is_var[:initial_size] = torch.tensor(pipeline.parser.is_var_mask, dtype=torch.bool, device=pipeline.device)

# Overwrite the pipeline arrays with the massive buffers
pipeline.parser.nodes = new_nodes
pipeline.parser.children = new_children
pipeline.parser.is_var_mask = new_is_var
# ---------------------------------------------

# Now the GPU has the buffer and the pointer it needs
resolvent = TorchSimpBatchPlus.compute_given_clause_resolvents_tensor(given_clause, processed_clause, pipeline)




# TorchSatSimpBatch.run_given_clause_benchmark_tensor(
#     tptp_strings=easier_tptp_strings,
#     pipeline=parser,
#     max_loops=2000
# )