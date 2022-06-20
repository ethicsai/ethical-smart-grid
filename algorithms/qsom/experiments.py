qsom_1 = {
    "name": "qsom_1",
    "q_learning_rate": 0.7,
    "q_discount_factor": 0.9,
    "update_all": True,
    "use_neighborhood": True,
    "sigma_state": 1.0,
    "lr_state": 0.8,
    "sigma_action": 1.0,
    "lr_action": 0.7,
    "initial_tau": 0.5,
    "tau_decay": False,
    "tau_decay_coeff": 1.0,
    "noise": 0.08
}

qsom_2 = {  # change: q_learning_rate
    "name": "qsom_2",
    "q_learning_rate": 0.1,
    "q_discount_factor": 0.9,
    "update_all": True,
    "use_neighborhood": True,
    "sigma_state": 1.0,
    "lr_state": 0.8,
    "sigma_action": 1.0,
    "lr_action": 0.7,
    "initial_tau": 0.5,
    "tau_decay": False,
    "tau_decay_coeff": 1.0,
    "noise": 0.08
}
qsom_3 = {  # change: q_discount_factor
    "name": "qsom_3",
    "q_learning_rate": 0.7,
    "q_discount_factor": 0.1,
    "update_all": True,
    "use_neighborhood": True,
    "sigma_state": 1.0,
    "lr_state": 0.8,
    "sigma_action": 1.0,
    "lr_action": 0.7,
    "initial_tau": 0.5,
    "tau_decay": False,
    "tau_decay_coeff": 1.0,
    "noise": 0.08
}
qsom_4 = {  # change: update_all
    "name": "qsom_4",
    "q_learning_rate": 0.7,
    "q_discount_factor": 0.9,
    "update_all": False,
    "use_neighborhood": True,
    "sigma_state": 1.0,
    "lr_state": 0.8,
    "sigma_action": 1.0,
    "lr_action": 0.7,
    "initial_tau": 0.5,
    "tau_decay": False,
    "tau_decay_coeff": 1.0,
    "noise": 0.08
}
qsom_5 = {  # change: use_neighborhood
    "name": "qsom_5",
    "q_learning_rate": 0.7,
    "q_discount_factor": 0.9,
    "update_all": True,
    "use_neighborhood": False,
    "sigma_state": 1.0,
    "lr_state": 0.8,
    "sigma_action": 1.0,
    "lr_action": 0.7,
    "initial_tau": 0.5,
    "tau_decay": False,
    "tau_decay_coeff": 1.0,
    "noise": 0.08
}
qsom_6 = {  # change: sigma_state
    "name": "qsom_6",
    "q_learning_rate": 0.7,
    "q_discount_factor": 0.9,
    "update_all": True,
    "use_neighborhood": True,
    "sigma_state": 0.1,
    "lr_state": 0.8,
    "sigma_action": 1.0,
    "lr_action": 0.7,
    "initial_tau": 0.5,
    "tau_decay": False,
    "tau_decay_coeff": 1.0,
    "noise": 0.08
}
qsom_7 = {  # change: lr_state
    "name": "qsom_7",
    "q_learning_rate": 0.7,
    "q_discount_factor": 0.9,
    "update_all": True,
    "use_neighborhood": True,
    "sigma_state": 1.0,
    "lr_state": 0.1,
    "sigma_action": 1.0,
    "lr_action": 0.7,
    "initial_tau": 0.5,
    "tau_decay": False,
    "tau_decay_coeff": 1.0,
    "noise": 0.08
}
qsom_8 = {  # change: sigma_action
    "name": "qsom_8",
    "q_learning_rate": 0.7,
    "q_discount_factor": 0.9,
    "update_all": True,
    "use_neighborhood": True,
    "sigma_state": 1.0,
    "lr_state": 0.8,
    "sigma_action": 0.1,
    "lr_action": 0.7,
    "initial_tau": 0.5,
    "tau_decay": False,
    "tau_decay_coeff": 1.0,
    "noise": 0.08
}
qsom_9 = {  # change: lr_action
    "name": "qsom_9",
    "q_learning_rate": 0.7,
    "q_discount_factor": 0.9,
    "update_all": True,
    "use_neighborhood": True,
    "sigma_state": 1.0,
    "lr_state": 0.8,
    "sigma_action": 1.0,
    "lr_action": 0.1,
    "initial_tau": 0.5,
    "tau_decay": False,
    "tau_decay_coeff": 1.0,
    "noise": 0.08
}

qsom_10 = {  # change: initial_tau
    "name": "qsom_10",
    "q_learning_rate": 0.7,
    "q_discount_factor": 0.9,
    "update_all": True,
    "use_neighborhood": True,
    "sigma_state": 1.0,
    "lr_state": 0.8,
    "sigma_action": 1.0,
    "lr_action": 0.7,
    "initial_tau": 0.1,
    "tau_decay": False,
    "tau_decay_coeff": 1.0,
    "noise": 0.08
}

qsom_11 = {  # change: tau_decay
    "name": "qsom_11",
    "q_learning_rate": 0.7,
    "q_discount_factor": 0.9,
    "update_all": True,
    "use_neighborhood": True,
    "sigma_state": 1.0,
    "lr_state": 0.8,
    "sigma_action": 1.0,
    "lr_action": 0.7,
    "initial_tau": 0.5,
    "tau_decay": True,
    "tau_decay_coeff": 1.0,
    "noise": 0.08
}

qsom_12 = {  # change: tau_decay_coef
    "name": "qsom_12",
    "q_learning_rate": 0.7,
    "q_discount_factor": 0.9,
    "update_all": True,
    "use_neighborhood": True,
    "sigma_state": 1.0,
    "lr_state": 0.8,
    "sigma_action": 1.0,
    "lr_action": 0.7,
    "initial_tau": 0.5,
    "tau_decay": True,
    "tau_decay_coeff": 0.1,
    "noise": 0.08
}

qsom_13 = {  # change: noise
    "name": "qsom_13",
    "q_learning_rate": 0.7,
    "q_discount_factor": 0.9,
    "update_all": True,
    "use_neighborhood": True,
    "sigma_state": 1.0,
    "lr_state": 0.8,
    "sigma_action": 1.0,
    "lr_action": 0.7,
    "initial_tau": 0.5,
    "tau_decay": False,
    "tau_decay_coeff": 1.0,
    "noise": 0.8
}
