import torch
import time

class UnifierEngine:
    """
    A version-agnostic developer API for the TorchATP ecosystem.
    Accepts any pipeline class (e.g., from TorchUnifyv2 or TorchUnifyCarlos)
    to allow instant hot-swapping inside Google Colab.
    """
    def __init__(self, pipeline_class, unifier_class, device=None, max_arity=3):
        """
        Args:
            pipeline_class: The ProverPipeline class to use.
            unifier_class: The BatchedGPUUnifier class to use.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"[{self.__class__.__name__}] Initializing engine on {self.device.type.upper()}")
        
        # Instantiate the injected pipeline version
        self.pipeline = pipeline_class(device=self.device, max_arity=max_arity)
        self.unifier_class = unifier_class
        
    def unify(self, string_pairs, use_tiered=False, fast_mode=False):
        """
        Unifies string pairs using the injected unifier class components.
        """
        if not string_pairs:
            return UnificationResult([], torch.empty(0), torch.empty(0), None, self.pipeline)
            
        self.pipeline.parser.clear()
        
        pair_indices = [
            self.pipeline.parser.parse_pair(left, right) 
            for left, right in string_pairs
        ]
            
        self.pipeline.last_run_standardized = False
        self.pipeline.batch_var_map_offset = 0
        
        pair_batch = torch.tensor(pair_indices, dtype=torch.long, device=self.device)
        
        # Instantiate the injected unifier version dynamically
        unifier = self.unifier_class(
            self.pipeline.parser.nodes, 
            self.pipeline.parser.children, 
            self.pipeline.parser.is_var_mask, 
            max_arity=self.pipeline.parser.max_arity
        )
        
        # Dynamically route to the method that exists on this specific class version
        if use_tiered and hasattr(unifier, 'tiered_unify'):
            subs, success_mask = unifier.tiered_unify(pair_batch)
        elif hasattr(unifier, '_unify_core'):
            subs, success_mask = unifier._unify_core(pair_batch, fast_mode=fast_mode)
        else:
            subs, success_mask = unifier.unify(pair_batch)
        
        return UnificationResult(string_pairs, subs, success_mask, unifier, self.pipeline)
    
class UnificationResult:
    """
    Encapsulates the output state of a unification wave.
    Provides utility methods to parse and display results intuitively.
    """
    def __init__(self, string_pairs, subs, success_mask, unifier, pipeline):
        self.string_pairs = string_pairs
        self.subs = subs
        self.success_mask = success_mask
        self.unifier = unifier
        self.pipeline = pipeline

    @property
    def metrics(self):
        """Returns a quick diagnostic tuple of successes."""
        total = len(self.success_mask)
        succeeded = int(self.success_mask.sum().item())
        return {"total": total, "succeeded": succeeded, "failed": total - succeeded}

    def print_summary(self):
        """Prints a highly readable, sanitized substitution summary."""
        # Reuses your custom decoder but wraps it beautifully
        self.pipeline.print_report(self.string_pairs, self.subs, self.success_mask, self.unifier)



class ColabBenchmarkSuite:
    """
    Automated benchmark runner tailored for interactive Colab cells.
    Handles device synchronization and presents a scannable performance grid.
    """
    def __init__(self, engine):
        self.engine = engine

    def run(self, base_pairs, sizes=[10, 100, 1000, 10000, 25000]):
        print("=" * 70)
        print(f"⚡ RUNNING BATCH SCALING BENCHMARK ON {self.engine.device.type.upper()} ⚡")
        print("=" * 70)
        
        # Warm up the hardware kernel to ensure accurate timings
        dummy = base_pairs * (10 // len(base_pairs) + 1)
        _ = self.engine.unify(dummy[:10], profile=False)
        if self.engine.device.type == "cuda":
            torch.cuda.synchronize()
            
        results_grid = []
        
        for size in sizes:
            # Inflate base template pairs to meet requested test density
            inflated = base_pairs * (size // len(base_pairs) + 1)
            test_set = inflated[:size]
            
            # Start strict timing bounds
            t_start = time.perf_counter()
            res = self.engine.unify(test_set, profile=False)
            if self.engine.device.type == "cuda":
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            
            elapsed_ms = (t_end - t_start) * 1000
            pairs_per_sec = size / (t_end - t_start)
            
            results_grid.append((size, elapsed_ms, pairs_per_sec))
            
        # Display a clean, professional Markdown table in the Colab log
        print("\n| Batch Size | Execution Time (ms) | Throughput (Pairs/Sec) |")
        print("|---|---|---|")
        for size, ms, throughput in results_grid:
            print(f"| {size:<10} | {ms:<19.2f} | {throughput:<20,.1f} |")
        print("\nBenchmark complete!")