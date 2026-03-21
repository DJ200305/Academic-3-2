import matplotlib.pyplot as plt

def load_results(path='tcp-qlearning-results.txt'):
    rows = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue

            # New format:
            # throughput total_sent total_energy_per_bit avg_energy_per_bit delivery_ratio drop_ratio
            # Optional old/labeled format:
            # PROTOCOL throughput total_sent ...
            if len(parts) >= 6:
                if parts[0].replace('.', '', 1).isdigit():
                    protocol = f"Run {len(rows)+1}"
                    nums = list(map(float, parts[:6]))
                else:
                    protocol = parts[0]
                    nums = list(map(float, parts[1:7]))
                rows.append({
                    "protocol": protocol,
                    "throughput_kbps": nums[0],
                    "total_sent_packets": nums[1],
                    "total_energy_per_bit_mj": nums[2],
                    "average_energy_per_bit_mj": nums[3],
                    "delivery_ratio": nums[4],
                    "drop_ratio": nums[5],
                })
    return rows

def plot_metric(rows, key, ylabel, title, out_file):
    labels = [r["protocol"] for r in rows]
    values = [r[key] for r in rows]

    if not values:
        print(f"No data for {key}")
        return

    plt.figure(figsize=(9, 6))
    colors = ['#C0C0C0' if 'jacob' in l.lower() else '#000000' for l in labels]
    bars = plt.bar(labels, values, color=colors, edgecolor='black', width=0.65)

    ymax = max(values) if max(values) > 0 else 1.0
    for bar, v in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + ymax * 0.02,
            f"{v:.4f}",
            ha='center',
            va='bottom',
            fontsize=9
        )

    plt.title(title, fontsize=13, fontweight='bold')
    plt.ylabel(ylabel, fontsize=11)
    plt.ylim(0, ymax * 1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    print(f"Saved: {out_file}")

def generate_all_graphs(path='tcp-qlearning-results.txt'):
    rows = load_results(path)
    if not rows:
        print("Error: No valid data found.")
        return

    plot_metric(rows, "throughput_kbps", "Throughput (Kbps)",
                "Throughput Comparison", "throughput_comparison.png")
    plot_metric(rows, "total_sent_packets", "Total Sent Packets",
                "Total Sent Packets Comparison", "sent_packets_comparison.png")
    plot_metric(rows, "total_energy_per_bit_mj", "Total Energy per Bit (mJ/bit)",
                "Total Energy per Bit Comparison", "total_energy_per_bit_comparison.png")
    plot_metric(rows, "average_energy_per_bit_mj", "Average Energy per Bit (mJ/bit)",
                "Average Energy per Bit Comparison", "average_energy_per_bit_comparison.png")
    plot_metric(rows, "delivery_ratio", "Delivery Ratio",
                "Delivery Ratio Comparison", "delivery_ratio_comparison.png")
    plot_metric(rows, "drop_ratio", "Drop Ratio",
                "Drop Ratio Comparison", "drop_ratio_comparison.png")

    plt.show()

if __name__ == "__main__":
    generate_all_graphs()
