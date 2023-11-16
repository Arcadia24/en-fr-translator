import argparse

import lightning as L
import torch
import torch.nn.functional as F
from books import HarryPotter
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from myTransformer import Transformer
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from utils import HarryPotterDataset, Vocab


class Lumiere(L.LightningModule):
    def __init__(self, vocab: Vocab, context_length: int, embed_size : int, n_head : int, num_layers : int, dropout : float) -> None:
        super().__init__()
        self.vocab_size = len(vocab)
        self.context_length = context_length
        self.model = Transformer(len(vocab), n_head, embed_size, context_length, dropout, num_layers)
        self.save_hyperparameters()
        self.vocab = vocab
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx) -> None:
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        logits = self.model(x)
        B, T, C = logits.size()
        loss = F.cross_entropy(logits.view(B * T, C), y.view(B * T))
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    @torch.inference_mode
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        logits = self.model(x)
        val_loss = F.cross_entropy(logits.view(-1, self.vocab_size), y.view(-1))
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    
    def on_validation_epoch_end(self) -> None:
        self.gen(torch.tensor([vocab.encode('Harry was missing ')]), 100)
        
    @torch.inference_mode
    def gen(self, ctx: torch.Tensor, max_seq_length: int) -> None:
        self.eval()
        print('Generation')
        print('----------')
        print(self.vocab.decode(ctx[0].tolist()), end='')
        for _ in range(max_seq_length):
            ctx = ctx.to(self.device)
            logits = self.model(ctx[:, -self.context_length:])
            logits = logits[:, -1, :]
            p = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(p, num_samples=1)
            ctx = torch.cat((ctx, idx_next), dim=1)
            print(self.vocab[idx_next.item()], end='')
        print()
        
    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=1e-3)
        scheduler = OneCycleLR(optimizer, max_lr=1e-2, epochs=1, steps_per_epoch=21000)
        return [optimizer], [scheduler]
            
            
if __name__ == "__main__":
    torch.cuda.empty_cache()
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--device', type=str, default='cuda')
    argparser.add_argument('--max_seq_length', type=int, default=1000)
    argparser.add_argument('--context_length', type=int, default=256)
    argparser.add_argument('--embed_size', type=int, default=390)
    argparser.add_argument('--n_head', type=int, default=6)
    argparser.add_argument('--n_layer', type=int, default=6)
    argparser.add_argument('--dropout', type=float, default=0.2)
    argparser.add_argument('--epochs', type=int, default=1)
    argparser.add_argument('--batch_size', type=int, default=64)
    argparser.add_argument('--lr', type=float, default=1e-3)
    args = argparser.parse_args()
    
    books = HarryPotter('harry_potter.txt')
    vocab = Vocab(books.content)
    
    train_set = HarryPotterDataset(args.context_length, books, vocab, train=True)
    val_set = HarryPotterDataset(args.context_length, books, vocab, train=False)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=11, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=11, shuffle=False, drop_last=False)
    
    model = Lumiere(vocab, args.context_length, args.embed_size, args.n_head, args.n_layer, args.dropout).to(args.device)
    trainfeature, trainlabel = next(iter(train_loader))

    logger = TensorBoardLogger("tb_logs", name="my_model")
    callbacks = [EarlyStopping(monitor="val_loss", mode="min"), ModelCheckpoint(dirpath='save/', monitor='val_loss', mode='min')]
    trainer = L.Trainer(
        callbacks= callbacks,
        max_epochs=args.epochs,
        val_check_interval=0.1,
        limit_val_batches=0.1,
        logger=logger,
        )
    trainer.fit(model, train_loader, val_loader)
    device = torch.device("cuda")
    print(device)
    idx = torch.tensor([vocab.encode('Harry was missing ')])
    model.gen(idx, 100)
    
    