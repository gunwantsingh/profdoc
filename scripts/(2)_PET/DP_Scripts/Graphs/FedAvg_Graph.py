import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the directory containing the files
directory = "../../DP_Scripts/Graphs/"

# List of file names pairs
file_pairs = [
    ("metrics_FedAvg.csv", "Simply_FedAvg.csv"),
    ("metrics_FaultTolerantFedAvg.csv", "Simply_FaultTolerantFedAvg.csv"),
    ("metrics_FedAdagrad.csv", "Simply_FedAdagrad.csv"),
    ("metrics_FedAdam.csv", "Simply_FedAdam.csv"),
    ("metrics_FedAvgM.csv", "Simply_FedAvgM.csv"),
    ("metrics_FedDP.csv", "Simply_FedDP.csv"),
    ("metrics_FedMedian.csv", "Simply_FedMedian.csv"),
    ("metrics_FedProx.csv", "Simply_FedProx.csv"),
    ("metrics_FedSecure.csv", "Simply_FedSecure.csv"),
    ("metrics_FedYogi.csv", "Simply_FedYogi.csv"),
    ("metrics_QFedAvg.csv", "Simply_QFedAvg.csv")
]

# Create a directory to save the PNG files if it doesn't exist
output_directory = "../../DP_Scripts/Graphs/Plots"
os.makedirs(output_directory, exist_ok=True)

# Iterate over each pair of file names
for file_pair in file_pairs:
    file1_name, file2_name = file_pair
    file1_path = os.path.join(directory, file1_name)
    file2_path = os.path.join(directory, file2_name)

    # Check if both files exist
    if os.path.exists(file1_path) and os.path.exists(file2_path):
        # Load data from the files
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)

        # Drop rows with NaN values in 'Aggregation Time' or 'Round Time' columns
        df1_filtered = df1.dropna(subset=['Aggregation Time', 'Round Time'])
        df2_filtered = df2.dropna(subset=['Aggregation Time', 'Round Time'])

        # Calculate the difference between 'Round Time' and 'Aggregation Time'
        df1_filtered['Difference'] = df1_filtered['Round Time'] - df1_filtered['Aggregation Time']
        df2_filtered['Difference'] = df2_filtered['Round Time'] - df2_filtered['Aggregation Time']

        # Plot the graph
        plt.plot(df1_filtered.index, df1_filtered['Difference'], label=file1_name)
        plt.plot(df2_filtered.index, df2_filtered['Difference'], label=file2_name)

        # Add labels and title
        plt.xlabel('Index')
        plt.ylabel('Difference (Round Time - Aggregation Time)')
        plt.title('Difference between Round Time and Aggregation Time')

        # Add legend
        plt.legend()

        # Save plot as PNG file
        output_file_path = os.path.join(output_directory, f"{file1_name.split('.')[0]}_{file2_name.split('.')[0]}.png")
        plt.savefig(output_file_path)

        # Clear the plot for the next iteration
        plt.clf()

print("Plots saved as PNG files.")
