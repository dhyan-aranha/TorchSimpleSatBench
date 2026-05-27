import time
import torch
import numpy as np

# Add TorchATP directory to sys.path for direct module import
workspace_dir = os.path.join(os.getcwd(), "TorchSimpleSatBench/TorchATP")
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

# --- Import User's Actual Pipeline ---
import TorchUnify

# --- Import PyRes Modules ---
import terms
from unification import mgu

# A diverse set of 10 unification problems
TEN_PAIRS = [
    ("p(X, a)", "p(b, Y)"),                  # 0: Simple multi-variable binding
    ("p(f(X), g(Y))", "p(f(a), g(b))"),      # 1: Deep structural binding
    ("f(X, X)", "f(a, Y)"),                  # 2: Intra-term dependency (X forces Y to be 'a')
    ("p(X)", "p(f(X))"),                     # 3: Immediate Occurs Check -> FAIL
    ("q(a, b)", "p(a, b)"),                  # 4: Predicate mismatch -> FAIL
    ("f(g(X), a)", "f(Y, X)"),               # 5: Cross-binding success (X=a, Y=g(a))
    ("h(X, Y)", "h(f(Y), g(X))"),            # 6: Recursive Occurs Check (X=f(g(X))) -> FAIL
    ("p(a, b, c)", "p(a, b, d)"),            # 7: Constant mismatch -> FAIL
    ("f(X, g(Y), Z)", "f(a, g(b), h(c))"),   # 8: Arity 3 success
    ("p(X)", "p(X)")                         # 9: Trivial self-unification
]

# =====================================================================
# 1. ACTUAL GPU PROVER PIPELINE (Using your TorchUnify.py)
# =====================================================================

class ActualGPUProver:
    def __init__(self, device='cuda'):
        self.device = device
        # max_arity=3 to handle formulas like p(a,b,c)
        self.pipeline = TorchUnify.NeuralProverPipeline(device=device, max_arity=3)
        
        # Parse all 10 templates into the global arena at once
        nodes, children, is_var, roots = self.pipeline.parser.parse_pairs(TEN_PAIRS)
        
        # Execute the Tensor Transition (Seal the arena)
        self.pipeline.parser.nodes = nodes.to(device)
        self.pipeline.parser.children = children.to(device)
        self.pipeline.parser.is_var_mask = is_var.to(device)
        
        # Store the 10 root pointer pairs as a Python list of tuples
        self.root_pairs = roots.tolist()
        
        # Track base size for cleanup
        self.initial_arena_size = self.pipeline.parser.nodes.shape[0]

    def reset_arena(self):
        """Prevents the torch.cat snowball from crashing RAM across iterations."""
        self.pipeline.parser.nodes = self.pipeline.parser.nodes[:self.initial_arena_size]
        self.pipeline.parser.children = self.pipeline.parser.children[:self.initial_arena_size]
        self.pipeline.parser.is_var_mask = self.pipeline.parser.is_var_mask[:self.initial_arena_size]

    def run_batched_tensor(self, batch_size):
        # Repeat the 10 pairs enough times to fill the requested batch size
        multiplier = (batch_size // len(self.root_pairs)) + 1
        pair_indices = (self.root_pairs * multiplier)[:batch_size]
        
        subs, success_mask, unifier = self.pipeline.prove_batch_indices(
            pair_indices, standardize_apart=False
        )
        
        successful_indices = torch.nonzero(success_mask).squeeze(-1)
        if successful_indices.dim() == 0:
            successful_indices = [successful_indices.item()]
        else:
            successful_indices = successful_indices.tolist()
            
        all_requests = []
        # Request to instantiate the left root for all successful unifications
        for idx in successful_indices:
            left_root = pair_indices[idx][0]
            all_requests.append([idx, left_root])
            
        if all_requests:
            req_tensor = torch.tensor(all_requests, dtype=torch.long, device=self.device)
            new_roots = self.pipeline.batched_instantiate_in_arena(req_tensor, unifier)
            return new_roots
            
        return []

    def verify_10_pairs(self):
        """Runs the 10 pairs and triggers the native TorchUnify print_report."""
        subs, success_mask, unifier = self.pipeline.prove_batch_indices(
            self.root_pairs, standardize_apart=False
        )
        
        # --- THE COLAB-ONLY FIX ---
        # Manually inject the missing attribute into the pipeline object 
        # so print_report doesn't crash when it looks for it!
        self.pipeline.last_run_standardized = False
        self.pipeline.batch_var_map_offset = 0
        # --------------------------
        
        # Use your custom built-in logger to decode the tensors
        self.pipeline.print_report(TEN_PAIRS, subs, success_mask, unifier)

# =====================================================================
# 2. REAL CPU SEQUENTIAL PIPELINE (Using Schulz's PyRes)
# =====================================================================

class PyResCPUProver:
    def __init__(self):
        # Pre-parse the 10 pairs
        self.parsed_pairs = []
        for l_str, r_str in TEN_PAIRS:
            self.parsed_pairs.append((terms.string2Term(l_str), terms.string2Term(r_str)))

    def unify_and_instantiate_single(self, index):
        s, t = self.parsed_pairs[index % len(self.parsed_pairs)]
        sigma = mgu(s, t)
        
        if sigma:
            return sigma(s)
        return None

    def run_sequential_loop(self, batch_size):
        resolvents = []
        for i in range(batch_size):
            res = self.unify_and_instantiate_single(i)
            resolvents.append(res)
        return resolvents

# =====================================================================
# 3. THE BENCHMARK EXECUTION SUITE
# =====================================================================

def run_colab_benchmark():
    print(f"PyTorch Version: {torch.__version__}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Selected Execution Device: {device.upper()}")
    print("=" * 70)

    batch_sizes = [10, 100, 1000, 10000, 25000]
    
    cpu_prover = PyResCPUProver()
    gpu_prover = ActualGPUProver(device=device)

    print(f"{'Batch Size':<12} | {'PyRes CPU (ms)':<22} | {'Your TorchUnify (ms)':<25} | {'Speedup Factor':<15}")
    print("-" * 75)

    for b_size in batch_sizes:
        gpu_prover.reset_arena()
        _ = gpu_prover.run_batched_tensor(b_size)
        if device == 'cuda': torch.cuda.synchronize()
        gpu_prover.reset_arena() 

        start_cpu = time.perf_counter()
        _ = cpu_prover.run_sequential_loop(b_size)
        end_cpu = time.perf_counter()
        cpu_time_ms = (end_cpu - start_cpu) * 1000

        start_gpu = time.perf_counter()
        _ = gpu_prover.run_batched_tensor(b_size)
        if device == 'cuda': torch.cuda.synchronize() 
        end_gpu = time.perf_counter()
        gpu_time_ms = (end_gpu - start_gpu) * 1000

        speedup = cpu_time_ms / gpu_time_ms if gpu_time_ms > 0 else 0
        print(f"{b_size:<12} | {cpu_time_ms:<22.2f} | {gpu_time_ms:<25.2f} | {speedup:.1f}x")

    print("=" * 70)
    
    # --- VERIFICATION PHASE ---
    print("\n\n" + "=" * 70)
    print("VERIFICATION PHASE: CPU vs GPU Output Equivalency")
    print("=" * 70)

    print("\n--- CPU PyRes Outputs ---")
    for i, (left_str, right_str) in enumerate(TEN_PAIRS):
        s = terms.string2Term(left_str)
        t = terms.string2Term(right_str)
        sigma = mgu(s, t)
        
        if sigma:
            # Resolves the fully unified string
            unified_str = terms.term2String(sigma(s))
            print(f"Pair {i}: SUCCESS -> Unified to: {unified_str}")
        else:
            print(f"Pair {i}: FAILED")

    print("\n--- GPU TorchUnify Detailed Report ---")
    gpu_prover.reset_arena()
    gpu_prover.verify_10_pairs()

if __name__ == "__main__":
    run_colab_benchmark()
