import torch
import torchaudio
from torch import nn
import gradio as gr


# Configuration
SAMPLE_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_kabyle_asr_optim.pt"

print("Device: ", DEVICE)


# -------- Tokenizer --------
class ModelTokenizer:
  def __init__(self, vocab):
    self.vocab = vocab
    self.idx2char = {i: c for i, c in enumerate(self.vocab)}
    self.blank_id = 0

  def decode(self, tokens):
    decoded = []
    prev_token = None
    for t in tokens:
      if t != self.blank_id and t != prev_token:
        decoded.append(self.idx2char.get(t, ''))
      prev_token = t
    return ''.join(decoded).strip()


# -------- Modèle --------
class ModelASR(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    self.encoder = bundle.get_model()
    self.fc = nn.Linear(bundle._params["encoder_embed_dim"], vocab_size)

  def forward(self, x):
    if x.dim() == 3 and x.shape[1] == 1:
      x = x.squeeze(1)
    feats, _ = self.encoder(x)
    return self.fc(feats)


# -------- Chargement du modèle --------
def load_model():
  checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
  vocab = checkpoint['vocab']
  model = ModelASR(len(vocab)).to(DEVICE)
  model.load_state_dict(checkpoint['model_state_dict'])
  model.eval()
  tokenizer = ModelTokenizer(vocab)
  print(f"✅ Modèle chargé depuis : {MODEL_PATH}")
  return model, tokenizer


# -------- Prétraitement audio --------
def load_audio(file_path):
  waveform, sr = torchaudio.load(file_path)
  if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)
  if sr != SAMPLE_RATE:
    waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
  waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-7)
  return waveform.to(DEVICE)


# -------- Prédiction --------
def transcribe(audio_file):
  model, tokenizer = load_model()
  waveform = load_audio(audio_file)
  with torch.no_grad():
    logits = model(waveform.unsqueeze(0))
    pred = torch.argmax(logits, dim=-1).squeeze(0).tolist()
    text = tokenizer.decode(pred)
  return text


# -------- Interface Gradio --------
def gradio_transcribe(audio):
  try:
    transcription = transcribe(audio)
    return transcription
  except Exception as e:
    return f"Erreur lors de la transcription : {str(e)}"


# Création de l'interface
iface = gr.Interface(
  fn=gradio_transcribe,
  inputs=gr.Audio(type="filepath", label="Uploader un fichier audio"),
  outputs=gr.Textbox(label="Transcription"),
  title="Reconnaissance Vocale (ASR) du Kabyle",
  description="Uploader un fichier MP3/WAV pour obtenir sa transcription",
  flagging_mode="never"
)

# -------- Main --------
if __name__ == "__main__":
  # Précharger le modèle au démarrage
  print("Préchargement du modèle...")
  load_model()

  # Lancer l'interface
  iface.launch(share=True)