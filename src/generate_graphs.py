import sys
import matplotlib.pyplot as plt
import pandas as pd

def plot_graph(log_file, y_label, title, output_file):
    data = pd.read_csv(log_file, sep=' ', header=None, names=['timestamp', 'value'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')

    plt.figure(figsize=(10, 5))
    plt.plot(data['timestamp'], data['value'])
    plt.xlabel('Time')
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 generate_graphs.py <cpu_log> <memory_log> <gpu_log>")
        sys.exit(1)

    cpu_log = sys.argv[1]
    memory_log = sys.argv[2]
    gpu_log = sys.argv[3]
    gpu_memory_log = sys.argv[4]

    plot_graph(cpu_log, 'CPU Usage (%)', 'CPU Usage Over Time', 'cpu_usage.png')
    plot_graph(memory_log, 'Memory Usage (MB)', 'Memory Usage Over Time', 'memory_usage.png')
    plot_graph(gpu_log, 'GPU Usage (%)', 'GPU Usage Over Time', 'gpu_usage.png')
    plot_graph(gpu_memory_log, 'GPU Memory Used (MB)', 'GPU Memory Usage Over Time', 'gpu_memory_usage.png')