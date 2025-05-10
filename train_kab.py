import torch
import torchaudio
import pandas as pd
import os
import unicodedata
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
import re
import atexit 
import requests
from torch.amp import GradScaler
from torch.amp import autocast


scaler = GradScaler(device="cuda")



# Configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

SAMPLE_RATE = 16000
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 3e-5
patience = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "best_kabyle_asr_optim.pt"
audios_path = 'audios'
csv_file = 'transcriptions.csv'

VALID_CHARS = "abcdefghijklmnopqrstuvwxyzàâçèéêëîïôùûüḍḥṣṭɣɛṛẓ-"
VALID_CHARS = re.escape(VALID_CHARS)
torch.backends.nnpack.enabled = False

# -------- Dataset --------
class KabyleDataset(Dataset):
    def __init__(self, audio_dir, csv_path):
        self.audio_dir = audio_dir
        self.df = pd.read_csv(csv_path, sep=';')
        self._validate_files()
        self.df['transcription'] = self._normalize_text(self.df['transcription'])
        print(f"✅ Dataset : {len(self.df)} échantillons valides")

    def _validate_files(self):
        valid_idx = []
        for idx, row in self.df.iterrows():
            audio_path = os.path.join(self.audio_dir, row['audio_file'])
            if (
                os.path.exists(audio_path)
                and audio_path.lower().endswith(('.wav', '.mp3'))
                and os.path.getsize(audio_path) >= 5 * 1024  # 5 KB
            ):
                valid_idx.append(idx)
            else:
                print(f"⚠️ Fichier audio ignoré (inexistant, mauvais format ou trop petit) : {row['audio_file']}")
        self.df = self.df.loc[valid_idx].reset_index(drop=True)


    def _normalize_text(self, texts):
        return (
          texts.str.lower()
          .apply(lambda x: unicodedata.normalize('NFC', x))
          .str.replace(fr'[^{VALID_CHARS}\s]', '', regex=True)
          .str.replace(r'\s+', ' ', regex=True)
          .str.strip()
               )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = os.path.join(self.audio_dir, self.df.iloc[idx]['audio_file'])
        try:
            waveform, sr = torchaudio.load(path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-7)
            if sr != SAMPLE_RATE:
                waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

            # Ajout de bruit aléatoire léger
            waveform += 0.005 * torch.randn_like(waveform)

            return waveform.squeeze(0), self.df.iloc[idx]['transcription']
    
        except Exception as e:
            print(f"❌ Erreur de lecture du fichier {path} : {e}")
            # Retourne un exemple vide ou neutre
        return torch.zeros(1), ""


# -------- Tokenizer --------
class KabyleTokenizer:
    def __init__(self, texts):
        chars = set("abcdefghijklmnopqrstuvwxyzàâçèéêëîïôùûüḍḥṣṭɣɛṛẓ-")
        extra = set(''.join(texts))
        self.vocab = ['<blank>', '<unk>'] + sorted(chars.union(extra))
        self.char2idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx2char = {i: c for i, c in enumerate(self.vocab)}
        print(f"🔤 Tokenizer : {len(self.vocab)} tokens")

    def encode(self, text):
        return [self.char2idx.get(c, 1) for c in text]

# -------- Modèle --------
class KabyleASR(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.encoder = bundle.get_model()
        # Fine-tuning complet
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(bundle._params["encoder_embed_dim"], vocab_size)

    def forward(self, x):
      # x shape: [B, 1, T] → squeeze to [B, T]
      if x.dim() == 3 and x.shape[1] == 1:
        x = x.squeeze(1)
      feats, _ = self.encoder(x)
      return self.fc(feats)


# -------- Collate --------
def collate_fn(batch):
    waveforms, texts = zip(*batch)
    waveforms = [w for w in waveforms if w.numel() > 0]
    texts = [t for t in texts if len(t.strip()) > 0]
    if len(waveforms) == 0:
        return torch.zeros(1, 16000).to(DEVICE), [""]  # placeholder pour éviter crash
    waveforms = nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
    return waveforms.to(DEVICE), texts

# -------- Entraînement --------
def train():
    dataset = KabyleDataset(audios_path, csv_file)
    tokenizer = KabyleTokenizer(dataset.df['transcription'])
    model = KabyleASR(len(tokenizer.vocab)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=0.5)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True, num_workers=40, prefetch_factor=4, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=40, prefetch_factor=4, persistent_workers=True)

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0

        for waveforms, texts in tqdm(train_loader, desc=f"🟢 Epoch {epoch+1}/{NUM_EPOCHS}"):
            optimizer.zero_grad()

            # Filtrage synchronisé
            cleaned = [(t, w) for t, w in zip(texts, waveforms) if len(t.strip()) > 0 and w.numel() > 0]
            if not cleaned:
                continue

            texts, waveforms = zip(*cleaned)
            waveforms = nn.utils.rnn.pad_sequence(waveforms, batch_first=True).to(DEVICE)

            encoded = [torch.tensor(tokenizer.encode(t), device=DEVICE) for t in texts]
            targets = torch.cat(encoded)
            target_lens = torch.tensor([len(e) for e in encoded], device=DEVICE)

            with autocast("cuda"):
                logits = model(waveforms)
                log_probs = torch.log_softmax(logits, dim=-1).permute(1, 0, 2)
                input_lens = torch.full((waveforms.size(0),), logits.size(1), dtype=torch.long, device=DEVICE)
                loss = criterion(log_probs, targets, input_lens, target_lens)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for waveforms, texts in val_loader:
                encoded = [torch.tensor(tokenizer.encode(t), device=DEVICE) for t in texts]
                targets = torch.cat(encoded)
                target_lens = torch.tensor([len(t) for t in encoded], device=DEVICE)
                logits = model(waveforms)
                log_probs = torch.log_softmax(logits, dim=-1).permute(1, 0, 2)
                input_lens = torch.full((logits.size(0),), logits.size(1), dtype=torch.long, device=DEVICE)
                val_loss += criterion(log_probs, targets, input_lens, target_lens).item()

        avg_train = total_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        scheduler.step(avg_val)
        print(f"📊 Epoch {epoch+1} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        if avg_val < best_loss:
            best_loss = avg_val
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab': tokenizer.vocab
            }, MODEL_SAVE_PATH)
            print(f"💾 Nouveau meilleur modèle sauvegardé (val_loss: {avg_val:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏹️ Early stopping déclenché")
                break

        torch.cuda.empty_cache()





# -------- Main --------
if __name__ == "__main__":
    @atexit.register  
    def goodbye(): 
        notif = "script_ended"
        url_notif = "https://ntfy.sh/" + notif
        y = requests.post(url_notif,  data=data.encode(encoding='utf-8'))
        print("Exiting Python Script!")
    mp.set_start_method("spawn", force=True)
    print("🚀 Démarrage de l'entraînement ASR Kabyle")
    print(f"📍 Appareil utilisé : {DEVICE}")
    try:
        train()
        if Path(MODEL_SAVE_PATH).exists():
            print(f"✅ Modèle sauvegardé : {MODEL_SAVE_PATH}")
        else:
            print("❌ Échec de la sauvegarde du modèle")
    except Exception as e:
        goodbye()
        print(f"Erreur lors de la transcription : {str(e)}")

