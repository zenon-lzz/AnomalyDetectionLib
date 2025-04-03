"""
=================================================
@Author: Zenon
@Date: 2025-03-26
@Description：Debugger File
==================================================
"""



if __name__ == '__main__':
    # %% md
    ## 4. Results Visualization
    # %%
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 5))
    plt.plot(test_energy)
    plt.axhline(y=threshold, color='r', linestyle='--', label='threshold')
    anomaly_indices = np.where(gt == 1)[0]
    plt.plot(np.arange(len(test_energy))[anomaly_indices], test_energy[anomaly_indices], 'r.', markersize=2,
             label='Anomaly')
    plt.title(f'{MODEL} Model Evaluation')
    plt.xlabel('TimeStamp')
    plt.ylabel('Anomaly Scores')
    plt.legend()
    plt.show()
