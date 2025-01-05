import pandas as pd
import matplotlib.pyplot as plt

def plot_fl_round_time(file1, file2):
    # Read CSV files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Extract FL round time (fourth column) from the first file, excluding NaN values
    fl_round_time1 = df1.iloc[:, 3].dropna().head(10)  # Extracting FL round time column from the fourth column for 10 FL rounds
    print("FL round time from first file:", fl_round_time1)

    # Extract FL round time (last column) from the second file, excluding NaN values
    fl_round_time2 = df2.iloc[:, -1].dropna().head(10)  # Extracting FL round time column from the last column for 10 FL rounds
    print("FL round time from second file:", fl_round_time2)

    # Generate index representing FL round
    index = range(1, len(fl_round_time1) + 1)

    # Plotting
    plt.plot(index, fl_round_time1, label='FedMedian_SMPC')
    plt.plot(index, fl_round_time2, label='FedMedian_No SMPC')
    plt.xlabel('FL Round')
    plt.ylabel('FL Round Time (seconds)')
    plt.title('Comparison of FL Round Time - FedMedian - SMPC and no SMPC')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    file1 = "/Users/gunwant/GCU_Main/_FL_Publication_2/_Homomorphic Encryption (HE + DP + SMPC + DFL)/SMPC/Metrics/FedMedian_smpc_metrics.csv"
    file2 = "/Users/gunwant/GCU_Main/_FL_Publication_2/_Homomorphic Encryption (HE + DP + SMPC + DFL)/SMPC/Metrics/FedMedian_simply_metrics.csv"
    plot_fl_round_time(file1, file2)





# import pandas as pd
# import matplotlib.pyplot as plt

# def plot_fl_round_time(file1, file2):
#     # Read CSV files
#     df1 = pd.read_csv(file1)
#     df2 = pd.read_csv(file2)

#     # Extract FL round time (last column)
#     fl_round_time1 = df1.iloc[:, -1].dropna()
#     fl_round_time2 = df2.iloc[:, -1].dropna()

#     # Generate index representing FL round
#     index1 = range(1, len(fl_round_time1) + 1)
#     index2 = range(1, len(fl_round_time2) + 1)

#     # Plotting
#     plt.plot(index1, fl_round_time1, label='FedAvg_SMPC')
#     plt.plot(index2, fl_round_time2, label='FedAvg_No SMPC')
#     plt.xlabel('FL Round')
#     plt.ylabel('FL Round Time (seconds)')
#     plt.title('Comparison of FL Round Time - FedAvg - SMPC and no SMPC')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# if __name__ == "__main__":
#     file1 = "/Users/gunwant/GCU_Main/_FL_Publication_2/_Homomorphic Encryption (HE + DP + SMPC + DFL)/SMPC/Metrics/FedAvg_smpc_metrics.csv"
#     file2 = "/Users/gunwant/GCU_Main/_FL_Publication_2/_Homomorphic Encryption (HE + DP + SMPC + DFL)/SMPC/Metrics/FedAvg_simply_metrics.csv"
#     plot_fl_round_time(file1, file2)
