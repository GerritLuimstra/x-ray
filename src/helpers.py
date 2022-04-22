import torch
from earlystopping import EarlyStopping

def obtain_predictions(model, dataloader, device, loss_fn):

    # Set the model into eval mode
    model.eval()

    # Obtain the predictions
    total_loss = 0
    y_preds = []
    y_trues = []

    with torch.no_grad():

        for (X, y) in dataloader:

            # Move the data to the device
            X = X.to(device)
            y = y.to(device).double()

            y_trues += list(y.detach().cpu().numpy())

            # Obtain the predictions
            y_pred = model(X).double().squeeze()

            # Accumulate the predictions
            y_preds += list((torch.round(y_pred) > 0.5).detach().cpu().numpy())

            # Accumulate the loss
            total_loss += loss_fn(y_pred, y)

    return total_loss, y_preds, y_trues

def train(model, device, optimizer, loss_fn, performance_funcs, training_dataloader, validation_dataloader, epochs=5, verbose=False, patience=10):

    # Setup the early stopping class
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    stop_early = False

    for epoch in range(epochs):

        if stop_early: 
            break

        # Loop over the batches
        for index, (X_train, y_train) in enumerate(training_dataloader):

            # Set the model in training mode
            model.train()

            # move the data to pytorch land
            X_train = X_train.to(device)
            y_train = y_train.to(device).double()

            # Obtain the predictions
            y_pred = model(X_train).double().squeeze()

            # Compute the loss
            # and keep track of it
            loss = loss_fn(y_pred, y_train)

            # Zero gradients and perform the backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():

                # Set the model in eval mode
                model.eval()

                # Obtain the validation performance
                valid_loss, valid_preds, y_valid = obtain_predictions(model, validation_dataloader, device, loss_fn)
                valid_performance = [func(y_valid, valid_preds) for func in performance_funcs]

                if (index + 1) % 10 == 0:

                    # Obtain the training data
                    train_loss, train_preds, y_train = obtain_predictions(model, training_dataloader, device, loss_fn)
                    train_performance = [func(y_train, train_preds) for func in performance_funcs]

                    print(f'epoch: {epoch}, iter: {index}/{len(training_dataloader)}, batch loss: {loss:.5f}, train_loss: {train_loss:.5f}, train_performance: {train_performance}, valid_loss: {valid_loss:.5f}, val_performance: {valid_performance}')
                else:
                    if verbose:
                        print(f'epoch: {epoch}, iter: {index}/{len(training_dataloader)}, batch loss: {loss:.5f}, train_loss: NA, train_performance: NA, valid_loss: {valid_loss:.5f}, val_performance: {valid_performance}')

                # Do we want to stop yet?
                early_stopping(valid_performance, model)
                if early_stopping.early_stop:
                    stop_early = True
                    break
    
    return (valid_performance, early_stopping.model) if not stop_early else (early_stopping.best_score, early_stopping.model)