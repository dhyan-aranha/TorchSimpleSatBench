import unittest
import torch
import sys
import os

# Ensure the local workspace is in the path if running from a sub-directory
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import TorchUnifyCarlos
import TorchUnifyPlus
# import TorchSat  # Uncomment this when you integrate the resolution tests

class TestGPUUnificationEdgeCases(unittest.TestCase):
    """
    Test suite for the highly-vectorized BatchedGPUUnifier.
    Validates pointer-chasing, race-condition prevention, and hardware-level occurs checks.
    """
    
    def setUp(self):
        # Set up a clean GPU Unification Pipeline for every single test
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Max arity 4 is required for the long-chain pointer test
        self.pipeline = TorchUnifyPlus.ProverPipeline(device=self.device, max_arity=4)
        self.capacity = 50_000 # Localized VRAM buffer strictly for testing

    def run_unify_pair(self, left_str, right_str):
        """
        Helper method to parse a pair, initialize the zero-copy VRAM arena, 
        and execute the unifier.
        """
        # 1. Parse the strings
        nodes, children, is_var, roots = self.pipeline.parser.parse_pairs([(left_str, right_str)])

        # 2. Emulate the Pre-Allocated Tensor Transition
        initial_size = nodes.shape[0]
        self.pipeline.parser.arena_ptr = initial_size

        new_nodes = torch.zeros(self.capacity, dtype=torch.long, device=self.device)
        new_children = torch.full((self.capacity, self.pipeline.parser.max_arity), -1, dtype=torch.long, device=self.device)
        new_is_var = torch.zeros(self.capacity, dtype=torch.bool, device=self.device)

        new_nodes[:initial_size] = nodes.to(self.device)
        new_children[:initial_size] = children.to(self.device)
        new_is_var[:initial_size] = is_var.to(self.device)

        self.pipeline.parser.nodes = new_nodes
        self.pipeline.parser.children = new_children
        self.pipeline.parser.is_var_mask = new_is_var

        # 3. Execute the batch
        pair_indices = roots.tolist()
        subs, success_mask, unifier = self.pipeline.prove_batch_indices(pair_indices, standardize_apart=False)

        return success_mask[0].item(), subs, unifier

    # --- THE TESTS ---

    def test_long_chain_pointer_chase(self):
        """Forces a cascading transitive binding (X1->X2->X3->X4->a)."""
        success, _, _ = self.run_unify_pair("f(X1, X2, X3, X4)", "f(X2, X3, X4, a)")
        self.assertTrue(success, "Long-Chain pointer chase failed to resolve completely.")

    def test_cross_wire_swap(self):
        """Tests sequential deferral (scatter) to prevent intra-wave race conditions."""
        success, _, _ = self.run_unify_pair("f(X, Y)", "f(Y, X)")
        self.assertTrue(success, "Cross-wire variable swap caused a cyclic race condition failure.")

    def test_delayed_occurs_check(self):
        """Ensures the inner BFS traces deep pointer history before approving bindings."""
        success, _, _ = self.run_unify_pair("h(X, g(X))", "h(Y, Y)")
        self.assertFalse(success, "Delayed Occurs-Check failed to catch the synthetic cycle.")

    def test_intra_wave_conflict(self):
        """Verifies deferred equations trigger structural mismatch fallback correctly."""
        success, _, _ = self.run_unify_pair("f(X, X)", "f(a, b)")
        self.assertFalse(success, "Intra-wave conflict failed to reject structurally mismatched constants.")

    def test_arity_padding_alignment(self):
        """Ensures -1 tensor padding aligns properly during horizontal logic masks."""
        success, _, _ = self.run_unify_pair("p(a, b, c)", "p(a, b, d)")
        self.assertFalse(success, "Padding alignment failed; incorrectly unified mismatched constants.")

class TestSaturationEdgeCases(unittest.TestCase):
    """
    Test suite for the Resolution and Factoring tensor kernels in TorchSat.
    Validates subsumption logic, empty-clause generation, and combinatorics.
    """
    
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pipeline = TorchUnifyPlus.ProverPipeline(device=self.device, max_arity=3)
        # Note: You can initialize your TorchSat NN or Dummy-NN here if needed

    def test_standardize_apart_trap(self):
        """
        Ensures variables in different clauses are mapped to distinct memory 
        coordinates before resolution, preventing false occurs-check failures.
        """
        c1 = "P(X)"
        c2 = "~P(f(X))"
        # TODO: Pass c1 and c2 into TorchSat.compute_batched_resolvents_between_tensors
        # self.assertTrue(derived_empty_clause)

    def test_multi_resolution_combinatorics(self):
        """
        Verifies the NxM cross-space matrix correctly expands and tests all 
        possible clashing literal pairs between two clauses.
        """
        c1 = "P(X) | P(Y)"
        c2 = "~P(a) | ~P(b)"
        # TODO: Assert that resolving c1 and c2 generates at least two distinct resolvents
        # e.g., P(Y) where X=a, and P(X) where Y=a

    def test_factoring_collision(self):
        """
        Verifies that unary factoring successfully clones and unifies internal positive literals.
        """
        c1 = "P(X, a) | P(b, Y)"
        # TODO: Pass c1 into TorchSat.compute_batched_factors_tensor
        # Assert that the output contains the simplified factor: P(b, a)

    def test_factoring_impossibility(self):
        """
        Ensures that an impossible internal unification gracefully returns an empty tensor.
        """
        c1 = "P(X, X) | P(a, b)"
        # TODO: Pass c1 into TorchSat.compute_batched_factors_tensor
        # Assert that the output list is empty (length 0)

    def test_subsumption_infinite_loop(self):
        """
        The ultimate test of Forward/Backward Subsumption filters.
        """
        c1 = "P(X)"
        c2 = "~P(Y) | P(f(Y))"
        # TODO: Pass c1 and c2 into run_neural_prover_tensor with a max generation cap.
        # Assert that the subsumption filter catches the explosive pattern, or that 
        # the engine hits the VRAM capacity exception safely without crashing the OS.


class TestArenaInstantiation(unittest.TestCase):
    """
    Test suite specifically for `batched_instantiate_in_arena`.
    Validates zero-copy VRAM allocation, 2D memoization deduplication, 
    and multi-batch reality isolation.
    """
    
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Increased to max_arity=4 to handle our test wrappers
        self.pipeline = TorchUnifyPlus.ProverPipeline(device=self.device, max_arity=4)
        self.capacity = 50_000
        
    def load_into_arena(self, nodes, children, is_var):
        """Helper to emulate the Pre-Allocated Tensor Transition."""
        initial_size = nodes.shape[0] if isinstance(nodes, torch.Tensor) else len(nodes)
        self.pipeline.parser.arena_ptr = initial_size

        new_nodes = torch.zeros(self.capacity, dtype=torch.long, device=self.device)
        new_children = torch.full((self.capacity, self.pipeline.parser.max_arity), -1, dtype=torch.long, device=self.device)
        new_is_var = torch.zeros(self.capacity, dtype=torch.bool, device=self.device)

        if initial_size > 0:
            new_nodes[:initial_size] = nodes.to(self.device)
            new_children[:initial_size] = children.to(self.device)
            new_is_var[:initial_size] = is_var.to(self.device)

        self.pipeline.parser.nodes = new_nodes
        self.pipeline.parser.children = new_children
        self.pipeline.parser.is_var_mask = new_is_var
        
        return initial_size

    def test_empty_requests_safety(self):
        """Ensures passing an empty request tensor doesn't crash or advance the VRAM pointer."""
        self.load_into_arena(
            torch.empty(0, dtype=torch.long), 
            torch.empty((0, self.pipeline.parser.max_arity), dtype=torch.long), 
            torch.empty(0, dtype=torch.bool)
        )
        
        initial_ptr = self.pipeline.parser.arena_ptr
        
        dummy_unifier = TorchUnifyPlus.BatchedGPUUnifier(
            self.pipeline.parser.nodes, self.pipeline.parser.children, 
            self.pipeline.parser.is_var_mask, max_arity=self.pipeline.parser.max_arity
        )
        dummy_unifier.subs = torch.empty((0, 0), device=self.device) 
        
        requests = torch.empty((0, 2), dtype=torch.long, device=self.device)
        new_roots = self.pipeline.batched_instantiate_in_arena(requests, dummy_unifier)
        
        self.assertEqual(new_roots, [], "Empty requests did not return an empty list.")
        self.assertEqual(self.pipeline.parser.arena_ptr, initial_ptr, "VRAM pointer advanced despite empty requests.")

    def test_basic_instantiation_and_pointer_advancement(self):
        """Verifies a standard substitution is written correctly and VRAM pointer advances."""
        # FIX: Use a wrapper to force shared variable scope
        strings = ["wrapper(f(X), f(a), p(X))"]
        nodes, children, is_var, roots = self.pipeline.parser.parse_clauses(strings)
        initial_size = self.load_into_arena(nodes, children, is_var)

        # Extract the shared pointers from the wrapper
        w_root = roots[0].item()
        f_X = self.pipeline.parser.children[w_root, 0].item()
        f_a = self.pipeline.parser.children[w_root, 1].item()
        p_X = self.pipeline.parser.children[w_root, 2].item()

        pair_batch = [[f_X, f_a]]
        subs, success_mask, unifier = self.pipeline.prove_batch_indices(pair_batch, standardize_apart=False)

        requests = torch.tensor([[0, p_X]], dtype=torch.long, device=self.device)
        new_roots = self.pipeline.batched_instantiate_in_arena(requests, unifier)

        self.assertGreater(self.pipeline.parser.arena_ptr, initial_size, "VRAM pointer did not advance.")

        r = new_roots[0].item()
        p_id = self.pipeline.parser.nodes[r].item()
        child_idx = self.pipeline.parser.children[r, 0].item()
        child_id = self.pipeline.parser.nodes[child_idx].item()

        reverse_vocab = {v: k for k, v in self.pipeline.parser.global_vocab.items()}
        self.assertEqual(reverse_vocab[p_id], "p")
        self.assertEqual(reverse_vocab[child_id], "a")

    def test_memoization_deduplication(self):
        """
        CRITICAL VRAM TEST: Ensures that if a variable appears twice, its 
        substituted target is only allocated ONCE in memory for that batch.
        """
        # FIX: Wrapper forces X to be the identical memory pointer everywhere
        strings = ["wrapper(f(X), f(g(a)), p(X, X))"]
        nodes, children, is_var, roots = self.pipeline.parser.parse_clauses(strings)
        self.load_into_arena(nodes, children, is_var)

        w_root = roots[0].item()
        f_X = self.pipeline.parser.children[w_root, 0].item()
        f_ga = self.pipeline.parser.children[w_root, 1].item()
        p_XX = self.pipeline.parser.children[w_root, 2].item()

        pair_batch = [[f_X, f_ga]]
        _, _, unifier = self.pipeline.prove_batch_indices(pair_batch, standardize_apart=False)

        requests = torch.tensor([[0, p_XX]], dtype=torch.long, device=self.device)
        new_roots = self.pipeline.batched_instantiate_in_arena(requests, unifier)

        r = new_roots[0].item()
        c1_idx = self.pipeline.parser.children[r, 0].item()
        c2_idx = self.pipeline.parser.children[r, 1].item()

        self.assertNotEqual(c1_idx, -1, "Child 1 pointer is missing.")
        self.assertEqual(c1_idx, c2_idx, "Memoization failed! Identical subtrees allocated at different memory addresses.")

    def test_multi_batch_reality_isolation(self):
        """
        Ensures that two different batches instantiating the exact same root 
        do not cross-contaminate their memory pointers or substitutions.
        """
        # FIX: Wrapper isolates memory logic
        strings = ["wrapper(f(X), f(a), f(b), p(X))"]
        nodes, children, is_var, roots = self.pipeline.parser.parse_clauses(strings)
        self.load_into_arena(nodes, children, is_var)

        w_root = roots[0].item()
        f_X = self.pipeline.parser.children[w_root, 0].item()
        f_a = self.pipeline.parser.children[w_root, 1].item()
        f_b = self.pipeline.parser.children[w_root, 2].item()
        p_X = self.pipeline.parser.children[w_root, 3].item()

        pair_batch = [[f_X, f_a], [f_X, f_b]]
        _, _, unifier = self.pipeline.prove_batch_indices(pair_batch, standardize_apart=False)

        requests = torch.tensor([[0, p_X], [1, p_X]], dtype=torch.long, device=self.device)
        new_roots = self.pipeline.batched_instantiate_in_arena(requests, unifier)

        self.assertEqual(len(new_roots), 2, "Expected 2 new roots for 2 requests.")
        r0 = new_roots[0].item()
        r1 = new_roots[1].item()

        c0_idx = self.pipeline.parser.children[r0, 0].item()
        c1_idx = self.pipeline.parser.children[r1, 0].item()

        id0 = self.pipeline.parser.nodes[c0_idx].item()
        id1 = self.pipeline.parser.nodes[c1_idx].item()

        reverse_vocab = {v: k for k, v in self.pipeline.parser.global_vocab.items()}
        
        self.assertEqual(reverse_vocab[id0], "a", "Batch 0 isolation failed.")
        self.assertEqual(reverse_vocab[id1], "b", "Batch 1 isolation failed.")
        
        self.assertNotEqual(c0_idx, c1_idx, "Batches critically cross-contaminated memory pointers!")



if __name__ == '__main__':
    unittest.main(verbosity=2)