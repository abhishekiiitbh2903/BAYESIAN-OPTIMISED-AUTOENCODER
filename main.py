from config import Config
from model.autoencoder import AutoEncoder
from utils import train_model
from dataset.script import get_dataloader
from utils import save_losses_to_csv,save_nmse_to_csv

if __name__ == "__main__":
    config = Config()
    model = AutoEncoder(config)
    total_params=0
    for p in model.parameters():
        total_params+=p.numel()
    print(f"Total Trainable Params in Model: {total_params}")
    train_loader, val_loader, test_loader = get_dataloader(config)
    train_losses, val_losses = train_model(model,train_loader=train_loader,val_loader=val_loader, config = config)
    save_losses_to_csv(train_losses, val_losses)
    save_nmse_to_csv(model, test_loader=test_loader, device=config.device)