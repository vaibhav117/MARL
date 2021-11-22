import argparse 

def parse_args(s=None):
    if s is None:
        parser= argparse.ArgumentParser(s)
    else:
        parser = argparse.ArgumentParser()

    # parser.add_argument('--project_name', default="fetchimage_asym", type=str) # wandb project name 
    parser.add_argument('--run_training_flag', default=True, type=bool)
    parser.add_argument('--num_knights', default=True, type=int)
    parser.add_argument('--num_archers', default=True, type=int)
    parser.add_argument('--device', default="cuda", type=str)

    args = parser.parse_args()
    return args