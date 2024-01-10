import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 


def ts_visualization(ts):
  plt.figure(figsize=(10, 6))
  plt.subplot(2, 1, 1)
  plt.title('Time Series', fontsize=18)
  plt.xlabel('Time', fontsize=18)
  plt.ylabel('Value', fontsize=18)
  plt.grid()
  plt.plot(ts, label='Time Series', color='blue')


def leadfollow_visualization(ts1,ts2):
  plt.figure(figsize=(15, 10))

  plt.subplot(2, 1, 1)
  plt.title('Leading Time Series (Adam Singing)', fontsize=22)
  plt.xlabel('Time', fontsize=22)
  plt.ylabel('Value', fontsize=22)
  plt.xticks(fontsize=18)
  plt.yticks(fontsize=18)
  plt.grid()
  plt.plot(ts1, label='Leading signal', color='blue')

  plt.subplot(2, 1, 2)
  plt.title('Following Time Series (Matt Singing)', fontsize=22)
  plt.xlabel('Time', fontsize=22)
  plt.ylabel('Value', fontsize=22)
  plt.xticks(fontsize=18)
  plt.yticks(fontsize=18)
  plt.grid()
  plt.plot(ts2, label='Following signal', color='red')

  plt.tight_layout()
  plt.show()


def alignment_plot(lead_ts, follow_ts, result):
    index_lead = result[2][1]
    index_follow = result[2][0]

    y1 = lead_ts[index_lead]
    y2 = follow_ts[index_follow]

    x3 = index_follow
    y3 = index_lead

    gap = len(y1) - len(y2)

    if gap < 0:  # leader shorter
        midrange = (max(y1) + min(y1)) / 2
        y1 = np.hstack([y1, np.ones(np.abs(gap)) * midrange])
        y3 = np.hstack([y3, np.ones(np.abs(gap)) * max(y3)])
    elif gap > 0:  # follower shorter
        midrange = (max(y2) + min(y2)) / 2
        y2 = np.hstack([y2, np.ones(np.abs(gap)) * midrange])
        x3 = np.hstack([x3, np.ones(np.abs(gap)) * max(x3)])

    standard_length = max(len(y1), len(y2))
    print(f"Standard Length: {standard_length}")

    x1 = np.linspace(0, 1, standard_length)
    x2 = np.linspace(0, 1, standard_length)

    fig = plt.figure(figsize=(10, 10))

    ax1 = fig.add_axes([0.05, 0.42, 0.3, 0.6])  
    ax2 = fig.add_axes([0.42, 0.05, 0.6, 0.3])  
    ax3 = fig.add_axes([0.42, 0.42, 0.6, 0.6])  

    ax1.plot(y1, x1)
    ax1.set_ylabel('Leader Time Series', fontsize=18)
    ax1.set_yticks([])  

    ax2.plot(x2, y2)
    ax2.set_xlabel('Following Time Series', fontsize=18)
    ax2.set_xticks([])  

    ax3.plot(y3, x3)

    plt.show()


def find_continuous_ranges(indices):
    ranges = []
    start = indices[0]
    for i in range(1, len(indices)):
        if indices[i] != indices[i - 1] + 1:
            ranges.append((start, indices[i - 1]))
            start = indices[i]
    ranges.append((start, indices[-1]))
    return ranges


def create_timeseries_from_array(data_array, start_date='20230601', periods=None, freq="min"):
    if periods is None:
        periods = len(data_array)
    dates = pd.date_range(start=start_date, periods=periods, freq="min")
    ts = pd.Series(data_array, index=list(range(len(data_array))))
    return ts


def plot_time_series_with_highlights(ts, highlight_indices, ax,
                                     title="Time Series with Highlighted Ranges", 
                                     ground_truth_indices=None):
    # Plot the time series data
    ax.plot(ts.index, ts.values, label='Time Series Data')

    # Highlight the predicted ranges
    if highlight_indices is not None:
        predicted_ranges = find_continuous_ranges(highlight_indices)
        for start_idx, end_idx in predicted_ranges:
            ax.axvspan(ts.index[start_idx], ts.index[end_idx],
                       color='green', alpha=1, label='Predicted')

    # Highlight the ground truth ranges
    if ground_truth_indices is not None:
        ground_truth_ranges = find_continuous_ranges(ground_truth_indices)
        for start_idx, end_idx in ground_truth_ranges:
            ax.axvspan(ts.index[start_idx], ts.index[end_idx],
                       color='red', alpha=1, label='Ground Truth')
            
    # Highlight overlap areas
    if highlight_indices is not None and ground_truth_indices is not None:
        overlap_indices = set(highlight_indices).intersection(ground_truth_indices)
        if overlap_indices:
            overlap_ranges = find_continuous_ranges(sorted(list(overlap_indices)))
            for start_idx, end_idx in overlap_ranges:
                ax.axvspan(ts.index[start_idx], ts.index[end_idx],
                           color='yellow', alpha=1, label='Overlap')

    # Adding labels and title for clarity
    ax.set_xlabel('Time Step', fontsize=18)
    ax.set_ylabel('Value', fontsize=18)
    ax.set_title(title, fontsize=20)

    # Create a legend with unique entries
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), fontsize=14)