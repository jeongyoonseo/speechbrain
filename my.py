import os
os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'

from speechbrain.inference.separation import SepformerSeparation
from speechbrain.utils.fetching import LocalStrategy
import torchaudio

# 모델 로드 (이미 다운로드 완료)
model = SepformerSeparation.from_hparams(
    source="speechbrain/sepformer-wham",
    savedir='pretrained_models/sepformer-wham',
    local_strategy=LocalStrategy.COPY
)

# 오디오 파일 분리 (2명의 음성이 섞인 파일)
est_sources = model.separate_file(path='mixed_audio.wav')

# 분리된 음성 저장
torchaudio.save("speaker1.wav", est_sources[:, :, 0].detach().cpu(), 8000)
torchaudio.save("speaker2.wav", est_sources[:, :, 1].detach().cpu(), 8000)

print("음성 분리 완료!")