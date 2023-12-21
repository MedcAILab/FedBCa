The server_classifiction.py contains the paths to the image data and label files, which can be used by modifying the paths. At the same time, server_classifiction.py contains the main function and many adjustable parameters.

The provided code from train_cls.py consists of two training functions for deep learning models:

train_step_cls:
   - Trains a classification model using a given training loader, optimizer, and criterion.
   - Handles GPU acceleration if available.
   - Displays a progress bar for each training epoch, tracking batch times and losses.
   - Supports model checkpointing by returning the learning rate, epoch loss, and the model's state dictionary.

train_step_cls_prox:
   - Extends the previous classification training with Federated Proximal (FedProx) optimization.
   - Introduces a hyperparameter `prox_mu` for controlling the proximal term.
   - Calculates the model parameter difference (`w_diff`) between the local model and a server model.
   - Augments the loss function with a proximal term to penalize large parameter deviations.
   - Performs training with the augmented loss and updates the model parameters accordingly.
   - Also displays a progress bar for tracking training progress.

Usage:
python server_ classifiction.py
