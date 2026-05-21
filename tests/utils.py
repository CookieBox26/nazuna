def set_training(model, training: bool):
    if training:
        model.train()
    else:
        model.eval()
