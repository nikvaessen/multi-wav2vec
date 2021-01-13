import pathlib
import fairseq

import torch as t
from fairseq.models.wav2vec import Wav2VecModel

from torchsummary import summary

root_dir = pathlib.Path(".").absolute().resolve()
model_path = root_dir / "models" / "wav2vec_small_960h.pt"

cp = t.load(str(model_path))
print([k for k in cp.keys()])

# args
print("### args ###")
for k, v in sorted(vars(cp['args']).items()):
    print(k, v)

# args = cp['args']
# args.prediction_steps = 1
# args.offset = 1
# args.activation = args.activation_fn

# model
print('### model ###')
for k, v in cp['model'].items():
    print(k)

print("")

# optimizer_history
print('### optimizer_history ###')
for v in cp['optimizer_history']:
    print(v)

print("")

# optimizer_history
print('### extra_state ###')
for k, v in cp['extra_state'].items():
    print(k)

print("")

# optimizer_history
print('### last_optimizer_state ###')
for k, v in cp['last_optimizer_state'].items():
    print(k)

# model = Wav2VecModel.build_model(args, task=None)
# model.load_state_dict(cp['model'], strict=False)
# model.eval()

cp = str(model_path)
model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp])
model = model[0]