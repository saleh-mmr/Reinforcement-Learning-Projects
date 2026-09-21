from model_train_test import ModelTrainTest


if __name__ == '__main__':
    # Parameters:
    train_mode = False
    render = not train_mode
    RL_hyperparams = {
        "train_mode": train_mode,
        "RL_load_path": './weights/weights_900.pth',
        "save_path": './weights/weights',
        "save_interval": 900,

        "clip_grad_norm": 5,
        "learning_rate": 1e-4,
        "discount_factor": 0.9,
        "batch_size": 128,
        "update_frequency": 20,
        "max_episodes": 900 if train_mode else 5,
        "max_steps": 200,
        "render": render,

        "epsilon_max": 1.0 if train_mode else -1,
        "epsilon_min": 0.02,
        "epsilon_decay": 0.999,

        "memory_capacity": 150_000,
        "render_fps": 60,
    }

    # Run
    DRL = ModelTrainTest(RL_hyperparams)  # Define the instance
    # Train
    if train_mode:
        DRL.train()
    else:
        # Test
        DRL.test(max_episodes=RL_hyperparams['max_episodes'])