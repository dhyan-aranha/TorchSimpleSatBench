import torch
import TorchUnify
import TorchSatSimpBatch
import TorchClauses 


given_clause_str = "cnf(c1, axiom, (~p(Z1, a) | ~p(Z1, X) | ~p(X, Z1)))."

target_clause_str = ["cnf(c2, axiom, (p(Z2, f(Z2)) | p(Z2, a)))."]

pipeline = TorchUnify.ProverPipeline()

given_clause = TorchClauses.parse_tptp_to_virtual_clause(given_clause_str, pipeline)

processed_pool = [TorchClauses.parse_tptp_to_virtual_clause(s, pipeline) for s in target_clause_str]

# Convert arrays to pytorch tensors. 
if isinstance(pipeline.parser.nodes, list):
    pipeline.parser.nodes = torch.tensor(
        pipeline.parser.nodes, dtype=torch.long, device=pipeline.device
    )
    pipeline.parser.children = torch.tensor(
        pipeline.parser.children, dtype=torch.long, device=pipeline.device
    )
    pipeline.parser.is_var_mask = torch.tensor(
        pipeline.parser.is_var_mask, dtype=torch.bool, device=pipeline.device
    )

TorchSatSimpBatch.compute_given_clause_resolvents_tensor(given_clause, processed_pool, pipeline)