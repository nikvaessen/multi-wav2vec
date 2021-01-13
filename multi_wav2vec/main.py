import pathlib as p
import torch as t

from fairseq.models.wav2vec import Wav2Vec2Model, Wav2VecConfig

root_dir = p.Path(".").absolute().resolve()
model_path = root_dir / "models" / "wav2vec_small_960h.pt"


state = t.load(str(model_path))

model = Wav2Vec2Model()
model.load_state_dict(state)
model.eval()

