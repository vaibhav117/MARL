import argparse 

def parse_args(s=None):
    if s is None:
        parser= argparse.ArgumentParser(s)
    else:
        parser = argparse.ArgumentParser()

    parser.add_argument('--project_name', default="Deep_Learning_Project", type=str)
    parser.add_argument('--experiment_name', default="Independent_DQNs", type=str)
    parser.add_argument('--experiment_dir', default="experiments", type=str)
    parser.add_argument('--run_id', default=None, type=str)                     # REQUIRED TO BE PASSED FOR FOR DEMOS
    parser.add_argument('--demo_len', default=20, type=int)                     # REQUIRED FOR FOR DEMOS

    parser.add_argument('--run_training_flag', default=True, type=bool)
    parser.add_argument('--device', default="cpu", type=str)

    parser.add_argument('--spawn_rate', default=1, type=int)
    parser.add_argument('--num_knights', default=1, type=int)
    parser.add_argument('--num_archers', default=0, type=int)
    parser.add_argument('--killable_knights', default=False, type=bool)
    parser.add_argument('--killable_archers', default=False, type=bool)
    parser.add_argument('--line_death', default=False, type=bool)

    parser.add_argument('--full_obs_width', default=300, type=int)
    parser.add_argument('--full_obs_height', default=180, type=int)
    parser.add_argument('--total_timesteps', default=1000000000, type=int)
    parser.add_argument('--replay_buffer_size', default=10000, type=int)
    parser.add_argument('--replay_start_size', default=1000, type=int)
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--eps_start', default=1., type=float)
    parser.add_argument('--eps_decay', default=.0001, type=float)
    parser.add_argument('--eps_min', default=0.15, type=float)
    parser.add_argument('--sync_target_network_freq', default=500, type=int)
    parser.add_argument('--network_update_freq', default=5, type=int)
    parser.add_argument('--reward_multiplier', default=100, type=int)


    

    args = parser.parse_args()
    return args