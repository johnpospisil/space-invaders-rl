"""
Quick test version of Phase 7 - runs 5 trials instead of 20.
Use this for rapid validation before full optimization.
"""

import sys
sys.path.insert(0, 'src')

from phase7_hyperparam_optimization import run_optimization, save_optimization_results, create_output_dirs

def main():
    """Run quick hyperparameter optimization."""
    print("\n" + "="*60)
    print("PHASE 7 QUICK TEST: HYPERPARAMETER OPTIMIZATION")
    print("="*60 + "\n")
    
    create_output_dirs()
    
    print("Quick Test Configuration:")
    print("  Trials: 5 (vs 20 full)")
    print("  Steps per trial: 200,000")
    print("  Expected time: ~50 minutes\n")
    
    # Run optimization with fewer trials
    study = run_optimization(n_trials=5, study_name='ppo_quick_opt')
    
    # Save results
    save_optimization_results(study)
    
    print("\n" + "="*60)
    print("QUICK TEST COMPLETE")
    print("="*60)
    print("\nNext: Run full optimization with phase7_hyperparam_optimization.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
