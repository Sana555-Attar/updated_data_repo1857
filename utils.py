from clearml import Task, Logger
import matplotlib.pyplot as plt
import numpy as np

def plot_graphs(history, string):
  
  train = history.history[string]
  valid = history.history['val_'+string]

  if string == 'accuracy':
      train_idx = np.argmax(train)
      train_best = train[train_idx]
      valid_idx = np.argmax(valid)
      valid_best = valid[valid_idx]

      Task.current_task().get_logger().report_single_value(name = "Best Train Acc", value=f'{train_best:.2f}')
      Task.current_task().get_logger().report_single_value(name = "Train Epoch#", value=train_idx+1)

      Task.current_task().get_logger().report_single_value(name = "Best Valid Acc", value=f'{valid_best:.2f}')
      Task.current_task().get_logger().report_single_value(name = "Valid Epoch#", value=valid_idx+1)
  

  if string == 'loss':
      train_idx = np.argmin(train)
      train_best = train[train_idx]
      valid_idx = np.argmin(valid)
      valid_best = valid[valid_idx]

      Task.current_task().get_logger().report_single_value("Lowest Train Loss", f'{train_best:.2f}')
      Task.current_task().get_logger().report_single_value("Train Epoch", train_idx+1)

      Task.current_task().get_logger().report_single_value("Lowest Valid Loss", f'{valid_best:.2f}')
      Task.current_task().get_logger().report_single_value("Valid Epoch#", valid_idx+1)

  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.title('Jarvis: Model '+ string)
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  #Task.current_task().get_logger().report_matplotlib_figure(string, "", plt)
  

def plot_confusion_matrix(actual, predicted, classes,
                          normalize=False,
                          title='Confusion matrix', figsize=(7,7),
                          cmap=plt.cm.Blues, path_to_save_fig=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(actual, predicted).T
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 1.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Predicted label')
    plt.xlabel('True label')
    Task.current_task().get_logger().report_matplotlib_figure(title, "", plt)
    
    if path_to_save_fig:
        plt.savefig(path_to_save_fig, dpi=300, bbox_inches='tight')