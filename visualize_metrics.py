import matplotlib.pyplot as plt
import os
import re

def parse_log_file(log_file):
    """
    Parse the log file to extract metrics for visualization.

    :param log_file: Path to the log file.
    :return: Dictionary containing parsed data (episodes, rewards, lengths, policy_losses, value_losses, entropies).
    """
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Log file not found: {log_file}")

    # Initialize storage for parsed data
    metrics = {
        "episodes": [],
        "rewards": [],
        "lengths": [],
        "policy_losses": [],
        "value_losses": [],
        "entropies": []
    }

    # Precompile regex patterns for efficiency
    patterns = {
        "episode": re.compile(r"Episode (\d+)"),  # Matches "Episode <number>"
        "reward": re.compile(r"Reward: ([\d.]+)"),  # Matches "Reward: <value>"
        "length": re.compile(r"Length: (\d+)"),  # Matches "Length: <number>"
        "policy_loss": re.compile(r"Policy Loss: ([\d\-.]+)"),  # Matches "Policy Loss: <value>"
        "value_loss": re.compile(r"Value Loss: ([\d\-.]+)"),  # Matches "Value Loss: <value>"
        "entropy": re.compile(r"Entropy: ([\d.]+)")  # Matches "Entropy: <value>"
    }

    current_episode = None  # Track the current episode being processed

    with open(log_file, "r") as file:
        for line in file:
            # Match episode number
            if match := patterns["episode"].search(line):
                current_episode = int(match.group(1))
                metrics["episodes"].append(current_episode)

            # Match reward
            elif match := patterns["reward"].search(line):
                metrics["rewards"].append(float(match.group(1)))

            # Match length
            elif match := patterns["length"].search(line):
                metrics["lengths"].append(int(match.group(1)))

            # Match policy loss
            elif match := patterns["policy_loss"].search(line):
                metrics["policy_losses"].append(float(match.group(1)))

            # Match value loss
            elif match := patterns["value_loss"].search(line):
                metrics["value_losses"].append(float(match.group(1)))

            # Match entropy
            elif match := patterns["entropy"].search(line):
                metrics["entropies"].append(float(match.group(1)))

    # Verify if episodes were parsed correctly
    if not metrics["episodes"]:
        raise ValueError("No episode data found in metrics. Cannot generate plots.")

    return metrics

def plot_metrics(metrics, save_dir=None):
    """
    Generate and save plots for training metrics.

    :param metrics: Dictionary containing metrics data (episodes, rewards, etc.).
    :param save_dir: Directory to save the plots (optional).
    """
    if not metrics["episodes"]:
        raise ValueError("No episode data found in metrics. Cannot generate plots.")

    # Plot each metric
    for key, values in metrics.items():
        if key == "episodes":  # Skip the episode index for plotting
            continue

        # Ensure x and y have the same length
        min_length = min(len(metrics["episodes"]), len(values))
        x_data = metrics["episodes"][:min_length]
        y_data = values[:min_length]

        plt.figure()
        plt.plot(x_data, y_data, label=key.replace("_", " ").capitalize(), linewidth=1.5)
        plt.title(f"{key.replace('_', ' ').capitalize()} over Episodes")
        plt.xlabel("Episodes")
        plt.ylabel(key.replace("_", " ").capitalize())
        plt.legend()
        plt.grid()

        # Save or show plot
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"{key}_plot.png"))
        else:
            plt.show()

def visualize_metrics(log_file=None, csv_file=None, save_dir=None):
    metrics = parse_log_file(log_file)
    print("Generating plots...")
    plot_metrics(metrics, save_dir)

# Example usage
if __name__ == "__main__":
    log_file_path = "logs/fpa_game_logs_20241205-044749.log"
    save_plots_directory = "plots"  # Directory to save plots
    visualize_metrics(log_file=log_file_path, save_dir=save_plots_directory)
    print("Visualization complete.")