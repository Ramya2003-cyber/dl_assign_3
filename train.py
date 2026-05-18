"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
import torch.nn.functional as F
from model import Transformer, make_src_mask, make_tgt_mask
from tqdm import tqdm
from lr_scheduler import NoamScheduler
import os
import argparse
import wandb
import torch.optim as optim

# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS  
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """


    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """
        log_probs = F.log_softmax(logits, dim=-1) 
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (self.vocab_size - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        pad_mask = (target == self.pad_idx)
        true_dist[pad_mask]=0.0
        loss=-torch.sum(true_dist*log_probs)
        non_pad_tokens=(~pad_mask).sum()
        with torch.no_grad():
            probs = torch.exp(log_probs)
            target_probs = probs.gather(1, target.unsqueeze(1)).squeeze(1)
            self.avg_confidence = target_probs[~pad_mask].mean().item()
        return loss/max(non_pad_tokens.item(),1)
    

        


# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP  
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch (float).

    """
    if is_train:
        model.train()
    else:
        model.eval()
    
    total_loss = 0.0
    total_acc = 0.0          
    total_conf = 0.0
    nbatches = 0
    pad_idx=getattr(loss_fn,'pad_idx',1)
    mode_desc="Train" if is_train else "Eval"
    pbar=tqdm(data_iter,desc=f"Epoch {epoch_num} [{mode_desc}]",leave=False)
    for batch in pbar:
        src,tgt=batch
        src=src.to(device)
        tgt=tgt.to(device)
        tgt_input=tgt[:,:-1]
        tgt_y=tgt[:,1:]
        src_mask=make_src_mask(src,pad_idx=pad_idx)
        tgt_mask=make_tgt_mask(tgt_input,pad_idx=pad_idx)
        with torch.set_grad_enabled(is_train):
            logits=model(src,tgt_input,src_mask,tgt_mask)
            logits_flat=logits.contiguous().view(-1,logits.size(-1))
            tgt_y_flat=tgt_y.contiguous().view(-1)
            loss=loss_fn(logits_flat,tgt_y_flat)
            preds = logits_flat.argmax(dim=-1)
            valid_mask = (tgt_y_flat != pad_idx)
            correct_tokens = ((preds == tgt_y_flat) & valid_mask).sum().item()
            total_valid_tokens = valid_mask.sum().item()
            acc = correct_tokens / max(total_valid_tokens, 1)
            total_acc += acc
            total_conf += getattr(loss_fn, 'avg_confidence', 0.0)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if epoch_num == 0 and nbatches < 1000:
                    # Dynamically find the first self-attention weights to avoid AttributeError
                    for name, param in model.encoder.layers[0].self_attn.named_parameters():
                        if 'weight' in name and param.grad is not None:
                            # Logs directly to W&B on a per-step basis!
                            import wandb
                            wandb.log({f"grad_norm_step": param.grad.norm().item()})
                            break # Only grab the first projection matrix (usually Query)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
        total_loss+=loss.item()
        nbatches+=1
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")
    run_epoch.last_acc = total_acc / max(nbatches, 1)
    run_epoch.last_conf = total_conf / max(nbatches, 1)
    return total_loss/max(nbatches,1)



# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING  
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.

    """
    # TODO: Task 3.3 — implement token-by-token greedy decoding
    model.eval()
    with torch.no_grad():
        memory=model.encode(src.to(device),src_mask.to(device))
    ys=torch.zeros(1,1,dtype=torch.long,device=device).fill_(start_symbol)
    for _ in range(max_len-1):
        tgt_mask=make_tgt_mask(ys,pad_idx=1).to(device)
        with torch.no_grad():
            dec_out=model.decode(memory,src_mask,ys,tgt_mask)
            logits=model.generator(dec_out[:,-1,: ])
        next_word = logits.argmax(dim=-1).item()
        next_tensor=torch.zeros(1,1,dtype=torch.long,device=device).fill_(next_word)
        ys=torch.cat([ys,next_tensor],dim=1)
        if next_word==end_symbol:
            break
    return ys


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION  
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
                          Each batch yields (src, tgt) token-index tensors.
        tgt_vocab       : Vocabulary object with idx_to_token mapping.
                          Must support  tgt_vocab.itos[idx]  or
                          tgt_vocab.lookup_token(idx).
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).

    """
    # TODO: Task 3 — loop test set, decode, compute and return BLEU
    model.eval()
    hypotheses=[]
    references=[]
    sos_idx=tgt_vocab.stoi.get('<sos>',2)
    eos_idx=tgt_vocab.stoi.get('<eos>',3)
    pad_idx=tgt_vocab.stoi.get('<pad>',1)
    pbar=tqdm(test_dataloader,desc="Evaluating BLEU",leave=False)
    with torch.no_grad():
        for batch in pbar:
            src,tgt=batch
            batch_size=src.size(0)
            for i in range(batch_size):
                single_src=src[i:i+1].to(device)
                src_mask=make_src_mask(single_src,pad_idx=pad_idx).to(device)
                ys=greedy_decode(model,single_src,src_mask,max_len=max_len,start_symbol=sos_idx,end_symbol=eos_idx,device=device)
                pred_token_ids=ys.squeeze(0).tolist()
                true_token_ids=tgt[i].tolist()
                pred_words=[tgt_vocab.itos[idx] for idx in pred_token_ids if idx not in (sos_idx,eos_idx,pad_idx)]
                true_words=[tgt_vocab.itos[idx] for idx in true_token_ids if idx not in (sos_idx,eos_idx,pad_idx)]
                hypotheses.append(pred_words)
                references.append([true_words])  # List of reference lists for corpus_bleu
    try:
        import bleu
        bleu_score=bleu.corpus_bleu(references,hypotheses)
        if bleu_score<1.0:
            bleu_score*=100
        return float(bleu_score)
    except Exception as e:
        print(f"Error computing BLEU: {e}")
        return float(bleu.compute_bleu(references,hypotheses)*100)



# ══════════════════════════════════════════════════════════════════════
# ❺  CHECKPOINT UTILITIES  (autograder loads your model from disk)
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.
    """
    


    # 2. Build the strict checkpoint dictionary according to the autograder keys
    checkpoint_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'model_config': model.model_config
    }

    # 3. Serialize and save the dictionary to disk safely
    print(f"Saving checkpoint state to '{path}' at the end of epoch {epoch}...")
    torch.save(checkpoint_dict, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Args:
        path      : Path to checkpoint file saved by save_checkpoint.
        model     : Uninitialised Transformer with matching architecture.
        optimizer : Optimizer to restore (pass None to skip).
        scheduler : Scheduler to restore (pass None to skip).

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).

    """
    # TODO: implement restore logic
    device=next(model.parameters()).device
    checkpoint=torch.load(path,map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch =checkpoint.get('epoch', 0)
    print(f"Checkpoint loaded from '{path}' (epoch {epoch}).")
    return int(epoch)


# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════


def run_training_experiment(args: argparse.Namespace) -> None:
    """
    Set up and run the full training experiment.

    Steps:
        1. Init W&B:   wandb.init(project="da6401-a3", config={...})
        2. Build dataset / vocabs from dataset.py
        3. Create DataLoaders for train / val splits
        4. Instantiate Transformer with hyperparameters from config
        5. Instantiate Adam optimizer (β1=0.9, β2=0.98, ε=1e-9)
        6. Instantiate NoamScheduler(optimizer, d_model, warmup_steps=4000)
        7. Instantiate LabelSmoothingLoss(vocab_size, pad_idx, smoothing=0.1)
        8. Training loop:
               for epoch in range(num_epochs):
                   run_epoch(train_loader, model, loss_fn,
                             optimizer, scheduler, epoch, is_train=True)
                   run_epoch(val_loader, model, loss_fn,
                             None, None, epoch, is_train=False)
                   save_checkpoint(model, optimizer, scheduler, epoch)
        9. Final BLEU on test set:
               bleu = evaluate_bleu(model, test_loader, tgt_vocab)
               wandb.log({'test_bleu': bleu})
    """
    # 1. Hyperparameter Configuration from CLI Arguments
    config = {
        "d_model": args.d_model,
        "N": args.N,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "warmup_steps": args.warmup_steps,
        "smoothing": args.smoothing,
        "max_len": args.max_len,
        "lr": args.lr, 
        "checkpoint_path": "best_model.pt",
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    # Initialize W&B Experiment Tracking
    wandb.init(project="da6401-a3", name=args.run_name, config=config,settings=wandb.Settings(console="off"))
    cfg = wandb.config
    device = cfg.device

    print(f"Running experiment on target device: {device}")

    # 2. Build Dataset and Vocabularies from dataset.py
    try:
        from dataset import Multi30kDataset
        print("Loading datasets and compiling vocabularies...")
        train_dataset = Multi30kDataset(split='train')
        val_dataset = Multi30kDataset(split='validation')
        test_dataset = Multi30kDataset(split='test')
        
        train_dataset.build_vocab(min_freq=2)
        val_dataset.de_vocab, val_dataset.en_vocab = train_dataset.de_vocab, train_dataset.en_vocab
        test_dataset.de_vocab, test_dataset.en_vocab = train_dataset.de_vocab, train_dataset.en_vocab
        
        de_vocab = train_dataset.de_vocab
        en_vocab = train_dataset.en_vocab
        train_dataset.process_data()
        val_dataset.process_data()
        test_dataset.process_data()
    except Exception as e:
        raise ImportError(f"Failed to import dataset.py: {e}")

    pad_idx = en_vocab.stoi.get('<pad>', 1)
    src_vocab_size = len(de_vocab)
    tgt_vocab_size = len(en_vocab)
    wandb.config.update({"src_vocab_size": src_vocab_size, "tgt_vocab_size": tgt_vocab_size})

    # 3. Create DataLoaders
    collate_fn = getattr(train_dataset, "collate_fn", None)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    # 4. Instantiate Transformer Model Shell
    model = Transformer(
        src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size,
        d_model=cfg.d_model, N=cfg.N, num_heads=cfg.num_heads,
        d_ff=cfg.d_ff, dropout=cfg.dropout
    ).to(device)

    model.model_config = {
        'src_vocab_size': src_vocab_size, 'tgt_vocab_size': tgt_vocab_size,
        'd_model': cfg.d_model, 'N': cfg.N, 'num_heads': cfg.num_heads,
        'd_ff': cfg.d_ff, 'dropout': cfg.dropout
    }

    # 5. Instantiate Optimizer and Scheduler
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model=cfg.d_model, warmup_steps=cfg.warmup_steps)

    # 6. Instantiate Custom Loss
    loss_fn = LabelSmoothingLoss(vocab_size=tgt_vocab_size, pad_idx=pad_idx, smoothing=cfg.smoothing)

    # 7. Core Training Loop
    print(f"Beginning training for {cfg.num_epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(cfg.num_epochs):
        train_loss = run_epoch(train_loader, model, loss_fn, optimizer, scheduler, epoch, is_train=True, device=device)
        train_acc = run_epoch.last_acc
        val_loss = run_epoch(val_loader, model, loss_fn, None, None, epoch, is_train=False, device=device)
        val_acc = run_epoch.last_acc     # Grab the sneaky attribute
        val_conf = run_epoch.last_conf
        current_lr = optimizer.param_groups[0]["lr"]
        wandb.log({
            "epoch": epoch, 
            "train_loss": train_loss, 
            "train_acc": train_acc,
            "val_loss": val_loss, 
            "val_acc": val_acc,
            "val_confidence": val_conf,
            "learning_rate": current_lr
        })
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}")
        save_checkpoint(model, optimizer, scheduler, epoch, path=cfg.checkpoint_path)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, path="best_model_weights.pt")

    # 8. Final BLEU Evaluation
    if os.path.exists("best_model_weights.pt"):
        checkpoint = torch.load("best_model_weights.pt", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
    test_bleu = evaluate_bleu(model, test_loader, tgt_vocab=en_vocab, device=device, max_len=cfg.max_len)
    wandb.log({'test_bleu': test_bleu})
    print(f"\nFINAL TEST BLEU SCORE: {test_bleu:.2f}")
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DA6401 Transformer Model")
    
    # Architecture params
    parser.add_argument("--d_model", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--N", type=int, default=6, help="Number of encoder/decoder layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feed-forward inner dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    
    # Training params
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=20, help="Total number of training epochs")
    parser.add_argument("--lr", type=float, default=1.0, help="Base learning rate multiplier for Noam") 
    parser.add_argument("--warmup_steps", type=int, default=4000, help="Noam scheduler warmup steps")
    parser.add_argument("--smoothing", type=float, default=0.1, help="Label smoothing epsilon")
    parser.add_argument("--max_len", type=int, default=100, help="Maximum generation length for BLEU eval")
    parser.add_argument("--run_name", type=str, default=None, help="Some run")
    
    # Execute the experiment with the parsed arguments
    args = parser.parse_args()
    run_training_experiment(args)
