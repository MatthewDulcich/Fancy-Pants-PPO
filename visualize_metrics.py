import pandas as pd
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
        "episode": re.compile(r"Starting episode (\d+)"),
        "reward": re.compile(r"Episode \d+ \| Reward: ([\d.]+) \| Length: (\d+)"),
        "policy_loss": re.compile(r"Policy Loss: ([\d\-.]+)"),
        "value_loss": re.compile(r"Value Loss: ([\d\-.]+)"),
        "entropy": re.compile(r"Entropy: ([\d.]+)")
    }

    with open(log_file, "r") as file:
        for line in file:
            # Match episode start
            if match := patterns["episode"].match(line):
                metrics["episodes"].append(int(match.group(1)))

            # Match reward and length
            if match := patterns["reward"].match(line):
                metrics["rewards"].append(float(match.group(1)))
                metrics["lengths"].append(int(match.group(2)))

            # Match policy loss
            if match := patterns["policy_loss"].match(line):
                metrics["policy_losses"].append(float(match.group(1)))

            # Match value loss
            if match := patterns["value_loss"].match(line):
                metrics["value_losses"].append(float(match.group(1)))

            # Match entropy
            if match := patterns["entropy"].match(line):
                metrics["entropies"].append(float(match.group(1)))

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
    """
    Parse log or CSV file and generate plots for visualization.

    :param log_file: Path to the log file (optional).
    :param csv_file: Path to the CSV file (optional).
    :param save_dir: Directory to save the plots (optional).
    """
    if not log_file and not csv_file:
        raise ValueError("Either log_file or csv_file must be provided.")

    # Parse data from the appropriate source
    if log_file:
        print(f"Parsing log file: {log_file}")
        metrics = parse_log_file(log_file)
    elif csv_file:
        print(f"Parsing CSV file: {csv_file}")
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        df = pd.read_csv(csv_file)
        metrics = {col.lower(): df[col].tolist() for col in df.columns}

    # Generate plots
    print("Generating plots...")
    plot_metrics(metrics, save_dir)


# Example usage
if __name__ == "__main__":
    log_file_path = "logs/fpa_game_logs_20241204-224138.log"  # Log file path
    csv_file_path = "Model Checkpoints/training_metrics.csv"  # CSV file path
    save_plots_directory = "plots"  # Directory to save plots

    # Call visualization
    visualize_metrics(log_file=log_file_path, save_dir=save_plots_directory)
    # OR
    # visualize_metrics(csv_file=csv_file_path, save_dir=save_plots_directory)