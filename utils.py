import matplotlib.pyplot as plt

def plot_predictions(df, column_pred, column_actual, title, ylabel):
    plt.figure(figsize=(10,5))
    plt.plot(df['timestamp'], df[column_pred], label='Predicted', marker='o')
    plt.plot(df['timestamp'], df[column_actual], label='Actual', marker='x')
    plt.xticks(rotation=45)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    return plt
