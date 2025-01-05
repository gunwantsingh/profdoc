import os
import pandas as pd
import matplotlib.pyplot as plt

# Load the specific CSV file
csv_file = 'Single_File.csv'
df = pd.read_csv(csv_file)

# Find columns with 'FL Round Time' in their names
fl_round_time_columns = [col for col in df.columns if 'FL Round Time' in col]

# Check if there are any matching columns
if fl_round_time_columns:
    # Plot the data for each 'FL Round Time' column
    plt.figure(figsize=(12, 8))

    for column in fl_round_time_columns:
        plt.plot(df.index, df[column], marker='o', linestyle='-', label=column)

    plt.title('FL Round vs. FL Round Time (HE & No HE)')
    plt.xlabel('FL Round')  # Assuming the index represents sequential rounds
    plt.ylabel('FL Round Time - HE (seconds)')
    plt.legend()
    plt.grid(True)

    # Save and show the plot
    # plt.savefig('FL Round Time Comparison.png')
    plt.show()
else:
    print(f"No columns with 'FL Round Time' found in {csv_file}.")
