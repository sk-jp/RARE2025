from sklearn.metrics import precision_recall_curve
import numpy as np

def ppv_at_recall(y_true, y_scores, recall_th=0.9):
    """
    y_true: ground truth labels (binary)
    y_scores: predicted scores (probabilities)
    recall_th: recall threshold to consider for precision calculation (0.0-1.0)
    """
    precision, recall, threshold = precision_recall_curve(y_true, y_scores)

    """
    print("Precision:", precision)
    print("Recall:", recall)
    print("th:", threshold)
    """

    """
    # Plotting the Precision-Recall curve (optional)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, marker='o', label='PR curve')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.legend()
    plt.show()
    """

    # Get indices where recall is greater than or equal to the threshold
    valid_idxs = np.where(recall >= recall_th)[0]
#    print("valid_idxs:", valid_idxs)

    if len(valid_idxs) == 0:
        # When no recall is above the threshold
        return None  
    else:
        # Select the precision values corresponding to valid indices
        idx = valid_idxs[-1]
        return precision[idx]

if __name__ == "__main__":
    # Example usage
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.7, 0.9, 0.95, 0.99, 0.85, 0.6])
    
    ppv = ppv_at_recall(y_true, y_scores, recall_th=0.5)
    print(f"PPV at recall threshold of 0.5: {ppv}")
    ppv = ppv_at_recall(y_true, y_scores, recall_th=0.9)
    print(f"PPV at recall threshold of 0.9: {ppv}")