def get_parameters():
    import argparse
    parser = argparse.ArgumentParser(description='Train a centralized model.')
    
    # Existing parameters...
    parser.add_argument('--dataset', type=str, default='physionet', help='Dataset name')
    parser.add_argument('--sample_tp', type=float, default=0.5, help='Sample time points')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset path')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    
    # NEW: Add solver parameters
    parser.add_argument('--solver_method', type=str, default='tsit5', 
                       choices=['euler', 'heun', 'dopri5', 'tsit5'],
                       help='Main ODE solver method')
    parser.add_argument('--encoder_solver_method', type=str, default='euler',
                       choices=['euler', 'heun', 'dopri5', 'tsit5'], 
                       help='Encoder ODE solver method')
    parser.add_argument('--use_jit', action='store_true', default=True,
                       help='Enable JIT compilation')
    parser.add_argument('--ode_rtol', type=float, default=1e-5,
                       help='ODE solver relative tolerance')
    parser.add_argument('--ode_atol', type=float, default=1e-7,
                       help='ODE solver absolute tolerance')
    
    return parser.parse_args()