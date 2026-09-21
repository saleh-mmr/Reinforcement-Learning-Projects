from RL.Cliff_Walking_DQN.trainer import Model_TrainTest

if __name__ == '__main__':
    # Parameters:
    train_mode = True
    render = not train_mode
    RL_hyperparams = {
        "train_mode"            : train_mode,
        "RL_load_path"          : './final_weights' + '_' + '800' + '.pth',
        "save_path"             : './final_weights',
        # "save_interval"         : 100,
        
        "clip_grad_norm"        : 4,
        "learning_rate"         : 1e-3,
        "discount_factor"       : 0.92,
        "batch_size"            : 32,
        "update_frequency"      : 40,
        "max_episodes"          : 800           if train_mode else 2,
        "max_steps"             : 200,
        "render"                : render,
        
        "epsilon_max"           : 0.999         if train_mode else -1,
        "epsilon_min"           : 0.01,
        "epsilon_decay"         : 0.994,
        
        "memory_capacity"       : 10_000        if train_mode else 0,
        
        "render_fps"            : 6,
        "sigma": 1.7e-9

    }
    
    
    # Run
    DRL = Model_TrainTest(RL_hyperparams) # Define the instance
    # Train
    if train_mode:
        DRL.train()
    else:
        # Test
        DRL.test(max_episodes = RL_hyperparams['max_episodes'])