from model_train_test import ModelTrainTest

if __name__ == '__main__':
    # Parameters:
    train_mode = True
    render = not train_mode
    map_size = 4  # 4x4 or 8x8
    RL_hyperparams = {
        "train_mode": train_mode,
        "RL_load_path": f'./4x4_weights/final_weights_2000.pth',
        "save_path": f'./4x4_weights/final_weights',
        "save_interval": 500,

        "clip_grad_norm": 3,
        "learning_rate": 6e-4,
        "discount_factor": 0.93,
        "batch_size": 32,
        "update_frequency": 10,
        "max_episodes": 4000 if train_mode else 5,
        "max_steps": 200,
        "render": render,

        "max_epsilon": 0.999 if train_mode else -1,
        "min_epsilon": 0.01,
        "epsilon_decay": 0.999,

        "memory_capacity": 4000 if train_mode else 0,

        "map_size": map_size,
        "num_states": map_size ** 2,
        "render_fps": 6,
        "weight_datafile_path": "conductance/datafile_V2.csv",
        "sigma": 1.7e-9
    }

    # Run
    DRL = ModelTrainTest(RL_hyperparams)  # Define the instance
    # Train
    if train_mode:
        DRL.train()
    else:
        # Test
        DRL.test(max_episodes=RL_hyperparams['max_episodes'])