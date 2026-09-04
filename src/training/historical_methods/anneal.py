"""

- temperature annealing functions
- making the activaiton gradually sharper during training
- not needed anymore

"""

import math


def get_annealing_factor(desired_num_iterations, desired_final_tau, starting_tau):
    """
    Returns the annealing factor for the SIGMOID_SMOOTHED temperature.
    """
    return -math.log(desired_final_tau / starting_tau) / desired_num_iterations


def compute_tau(iterations, starting_tau, annealing_factor):
    """
    Returns the annealed temperature for the SIGMOID_SMOOTHED.
    """
    return starting_tau * math.exp(-annealing_factor * iterations)


# def get_run_annealing_factor(config):
#     """
#     get the annealing factor for the current run, the number of iterations is computed from the config; each iteration is a batch
#     """
#     dataset = get_dataset_obj(config["dataset"], "TRAIN")
#     desired_num_iterations = config["epochs"] * (len(dataset) // config["batch_size"])
#     desired_final_tau, starting_tau = config["desired_final_tau"], config["tau"]
#     return get_annealing_factor(desired_num_iterations, desired_final_tau, starting_tau)


# def init(config, runs_dir: Path, run_name=None):
#     """
#     dev mode: use a small subset of the dataset for faster debugging
#     """
#     # # ---------------- setting up the logger ----------------
#     # if config.get("masking_mechanism") == "SIGMOID_SMOOTHED":
#     #     config.update({"annealing_factor": get_run_annealing_factor(config)})


# annealing_factor = run_loader.config.get("annealing_factor", 1.0)
