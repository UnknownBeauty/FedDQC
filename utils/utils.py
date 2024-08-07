import math

def cosine_learning_rate(current_round, total_rounds, initial_lr=0.001, min_lr=0):
    """
    Compute the learning rate based on a cosine schedule.

    :param current_round: The current training round (0-indexed).
    :param total_rounds: The total number of training rounds.
    :param initial_lr: The initial learning rate.
    :param min_lr: The minimum learning rate.
    :return: The computed learning rate for the current round.
    """
    # Compute the cosine learning rate
    cosine_lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * current_round / total_rounds))
    return cosine_lr

def plot_learning_rate(lr_list, num_rounds, initial_lr, min_lr):
    """
    Plot the learning rate schedule.

    :param lr_list: The list of learning rates for each round.
    :param num_rounds: The total number of training rounds.
    :param initial_lr: The initial learning rate.
    :param min_lr: The minimum learning rate.
    """
    import matplotlib.pyplot as plt

    # Plot the learning rate schedule
    plt.figure(figsize=(8, 6))
    plt.plot(range(num_rounds), lr_list, label="Cosine Learning Rate Schedule")
    plt.axhline(y=initial_lr, color='r', linestyle='--', label="Initial Learning Rate")
    plt.axhline(y=min_lr, color='g', linestyle='--', label="Minimum Learning Rate")
    plt.xlabel("Training Round")
    plt.ylabel("Learning Rate")
    plt.title("Cosine Learning Rate Schedule")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('figs/cosine_learning_rate_schedule.png')

if __name__ == "__main__":

    # Example usage:
    num_rounds = 300
    initial_lr = 5e-5
    min_lr = 1e-6

    lr_list = []
    for round in range(num_rounds):
        lr = cosine_learning_rate(round, num_rounds, initial_lr, min_lr)
        lr_list.append(lr)
        print(f"Round {round + 1}/{num_rounds}, Learning Rate: {lr:.8f}")
    
    plot_learning_rate(lr_list, num_rounds, initial_lr, min_lr)

