import pandas as pd
import matplotlib.pyplot as plt

def plot_fl_round_time(file1, file2):
    # Read CSV files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Extract FL round time (last column)
    fl_round_time1 = df1.iloc[:, -1].dropna()
    fl_round_time2 = df2.iloc[:, -1].dropna()

    # Generate index representing FL round
    index1 = range(1, len(fl_round_time1) + 1)
    index2 = range(1, len(fl_round_time2) + 1)

    # Plotting
    plt.plot(index1, fl_round_time1, label='FedDP_HE')
    plt.plot(index2, fl_round_time2, label='FedDP_No HE')
    plt.xlabel('FL Round')
    plt.ylabel('FL Round Time (seconds)')
    plt.title('Comparison of FL Round Time - FedDP - HE and No HE')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    file1 = "/Users/gunwant/GCU_Main/_FL_Publication_2/_Homomorphic Encryption (HE + DP + SMPC + DFL)/HE_only/10-FedDP/FedDP_HEOnly_metrics.csv"
    file2 = "/Users/gunwant/GCU_Main/_FL_Publication_2/_Homomorphic Encryption (HE + DP + SMPC + DFL)/HE_only/10-FedDP/FedDP_Simply_metrics.csv"
    plot_fl_round_time(file1, file2)
