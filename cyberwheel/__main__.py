import sys
from cyberwheel.utils import parse_default_override_args, parse_eval_override_args, parse_override_args, parse
from cyberwheel.runners import train_cyberwheel, evaluate_cyberwheel, run_cyberwheel, run_visualization_server, train_table_agents

def display_help():
    sys.argv = ['']
    print("---------------------------------------------------------------------------------------------------\nTraining Cyberwheel:\n\n")
    parse_override_args(print_help=True)
    print("---------------------------------------------------------------------------------------------------\nEvaluating Cyberwheel:\n\n")
    parse_eval_override_args(print_help=True)
    print("---------------------------------------------------------------------------------------------------\nRunning Cyberwheel:\n\n")
    parse_default_override_args(print_help=True)

if __name__ == "__main__":
    """
    Usage:
        python -m cyberwheel <mode> <config(s)>
    
    Example:
        python -m cyberwheel train train_config.yaml
        python -m cyberwheel evaluate eval_config.yaml
        python -m cyberwheel emulate eval_config.yaml
        python -m cyberwheel run run_config.yaml
        python -m cyberwheel train_expansion expansion_config.yaml
        python -m cyberwheel train_all train_first_config.yaml train_second_config.yaml
    """
    if len(sys.argv) > 2:
        mode = sys.argv.pop(1)
        if mode == 'train_all' and len(sys.argv) == 3:
            # husket å fjerne action logs og modeller du ikke vil ha?
            config1 = sys.argv.pop(1)
            config2 = sys.argv.pop(1)
            print(f"Training with configs: {config1} and {config2}")
        else:
            config = sys.argv.pop(1)
            args = parse(config, mode) if mode in ['train', 'train_expansion', 'evaluate', 'emulate', 'run'] else None
        
        if mode == 'visualizer':
            run_visualization_server(config)
            sys.exit(0)

        if mode == 'train':
            train_cyberwheel(args)
        elif mode == 'train_expansion':
            train_table_agents(args)
        elif mode == 'train_all':
            args = parse(config1, mode)
            train_table_agents(args)
            args = parse(config2, mode)
            train_cyberwheel(args)
        elif mode == 'evaluate':
            evaluate_cyberwheel(args, emulate=False)
        elif mode == 'emulate':
            evaluate_cyberwheel(args, emulate=True)
        elif mode == 'run':
            run_cyberwheel(args)
        else:
            display_help()