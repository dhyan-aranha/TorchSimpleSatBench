import torch
import TorchUnify

def test_occurs_check_cycle():
    print("Initializing Neural Prover Pipeline...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize the pipeline
    pipeline = TorchUnify.NeuralProverPipeline(device=device, max_arity=2)
    
    # The notorious cyclic test case
    test_pairs = [
        ("f(X, Y)", "f(g(Y), g(X))")
    ]
    
    print("=" * 60)
    print(f"Running Cyclic Occurs-Check Test on {device.upper()}")
    print("=" * 60)
    
    # CRITICAL: We set standardize_apart=False. 
    # If we standardized apart, it would become f(X1, Y1) = f(g(Y2), g(X2)) 
    # which is perfectly valid and does not cycle!
    subs, success_mask, unifier = pipeline.prove_batch(
        test_pairs, 
        standardize_apart=False
    )
    
    # Trigger your native built-in report decoder
    pipeline.print_report(test_pairs, subs, success_mask, unifier)
    
    # Explicitly verify the architectural result
    print("\n" + "=" * 60)
    if not success_mask[0].item():
        print("✅ ARCHITECTURE TEST PASSED: The GPU correctly deferred the")
        print("   variables and successfully FAILED the Occurs Check!")
    else:
        print("❌ ARCHITECTURE TEST FAILED: The GPU encountered a race")
        print("   condition and incorrectly reported a false SUCCESS.")
    print("=" * 60)

if __name__ == "__main__":
    test_occurs_check_cycle()