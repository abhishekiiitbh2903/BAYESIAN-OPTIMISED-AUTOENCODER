import csv
import torch
import torch.nn as nn 
from tqdm import tqdm
import numpy as np
import gc

def save_losses_to_csv(train_losses, val_losses, file_path="loss_log.csv"):
    assert len(train_losses) == len(val_losses), "Mismatch in length of loss lists."

    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Validation Loss"])
        for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), start=1):
            writer.writerow([epoch, train_loss, val_loss])

    print(f" Losses saved to {file_path}")

    return


def train_model(model, train_loader, val_loader, config):
    model = model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    def lr_lambda(step):
        return config.decay_rate ** (step // config.decay_steps)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    global_step = 0
    last_printed_lr = optimizer.param_groups[0]['lr']

    for epoch in range(config.num_epochs):
        use_lbfgs = epoch >= config.num_epochs - 2
        running_train_loss = 0.0

        if not use_lbfgs:
            model = model.to(config.device)
            model.train()
            train_bar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{config.num_epochs}] Training", dynamic_ncols=True)

            for batch in train_bar:
                inputs = batch.to(next(model.parameters()).device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, inputs)

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                loss.backward()
                optimizer.step()
                scheduler.step()

                loss_value = loss.item()
                running_train_loss += loss_value
                train_bar.set_postfix(Step=global_step, Loss=f"{loss_value:.4f}", LR=f"{scheduler.get_last_lr()[0]:.6f}")
                global_step += 1

                current_lr = scheduler.get_last_lr()[0]
                if current_lr != last_printed_lr:
                    print(f"Step {global_step}: Learning rate changed to {current_lr:.6f}")
                    last_printed_lr = current_lr

        else:
            print(f"Running L-BFGS on CPU at epoch {epoch+1}")
            try:
                model = model.to("cpu")
                torch.cuda.empty_cache()
                gc.collect()

                full_inputs = torch.cat([batch.to("cpu") for batch in train_loader], dim=0)

                current_lr = optimizer.param_groups[0]['lr'] if optimizer else config.learning_rate
                optimizer = torch.optim.LBFGS(
                    model.parameters(), lr=current_lr, max_iter=20, history_size=10, line_search_fn="strong_wolfe"
                )

                def closure():
                    optimizer.zero_grad()
                    outputs = model(full_inputs)
                    loss = criterion(outputs, full_inputs)

                    if torch.isnan(loss) or torch.isinf(loss):
                        raise ValueError("Invalid loss detected during L-BFGS closure.")

                    loss.backward()
                    return loss

                loss = optimizer.step(closure)
                loss_value = loss.detach().item() if loss.requires_grad else loss.item()
                running_train_loss = loss_value
                print(f"L-BFGS step completed with loss: {loss_value:.4f}")

            except RuntimeError as e:
                print("L-BFGS failed due to memory or numerical instability.")
                print(f"Error: {e}")
                print("Reverting back to Adam.")
                model = model.to(config.device)
                torch.cuda.empty_cache()
                gc.collect()
                current_lr = optimizer.param_groups[0]['lr'] if optimizer else config.learning_rate
                optimizer = torch.optim.Adam(model.parameters(), lr=current_lr)
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
                continue

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        running_val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f"[Epoch {epoch+1}/{config.num_epochs}] Validation", dynamic_ncols=True)

        with torch.no_grad():
            for batch in val_bar:
                inputs = batch.to(next(model.parameters()).device)
                outputs = model(inputs)
                loss = criterion(outputs, inputs)

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                running_val_loss += loss.item()
                val_bar.set_postfix(Loss=f"{loss.item():.4f}")

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if not use_lbfgs:
            print(f"Epoch {epoch+1}/{config.num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        else:
            print(f"Epoch {epoch+1}/{config.num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    return train_losses, val_losses

def compute_avg_nmse_db(model, test_loader, snr, device):
    model.eval()
    model.to(device)

    total_num = 0.0
    total_den = 0.0

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch,snr_db=snr)

            num = torch.norm(batch - output) ** 2
            den = torch.norm(batch) ** 2

            total_num += num.item()
            total_den += den.item()

    avg_nmse = total_num / total_den
    nmse_db = 10 * np.log10(avg_nmse) 
    return nmse_db


def save_nmse_to_csv(model, test_loader, device):
    results = []
    for snr_value in range(1, 16):
        nmse_db = compute_avg_nmse_db(model, test_loader, snr=snr_value, device=device)
        results.append((snr_value, nmse_db))
        print(f"Average NMSE at SNR={snr_value} dB: {nmse_db:.2f} dB")

    csv_file = "snr_vs_nmse_59.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["SNR (dB)", "NMSE (dB)"])
        for snr_value, nmse_db in results:
            writer.writerow([snr_value, nmse_db])

    print(f"\n✅ NMSE results saved to: {csv_file}")