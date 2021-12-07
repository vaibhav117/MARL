import argparse 

def parse_args(s=None):
    if s is None:
        parser= argparse.ArgumentParser(s)
    else:
        parser = argparse.ArgumentParser()

    parser.add_argument('--project_name', default="Deep_Learning_Project", type=str)
    parser.add_argument('--experiment_name', default="Independent_DQNs", type=str)

    parser.add_argument('--run_training_flag', default=True, type=bool)
    parser.add_argument('--device', default="cuda", type=str)

    parser.add_argument('--num_knights', default=0, type=int)
    parser.add_argument('--num_archers', default=1, type=int)
    parser.add_argument('--killable_knights', default=False, type=bool)
    parser.add_argument('--killable_archers', default=False, type=bool)
    parser.add_argument('--line_death', default=False, type=bool)

    parser.add_argument('--total_timesteps', default=2000000, type=int)
    parser.add_argument('--replay_buffer_size', default=10000, type=int)
    parser.add_argument('--replay_start_size', default=10000, type=int)
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--eps_start', default=1., type=float)
    parser.add_argument('--eps_decay', default=.999985, type=float)
    parser.add_argument('--eps_min', default=0.02, type=float)
    parser.add_argument('--sync_target_network_freq', default=1000, type=int)
    parser.add_argument('--network_update_freq', default=100, type=int)


    

    args = parser.parse_args()
    return args