from matplotlib import pyplot as plt
from keras.models import load_model
import numpy as np
import os
import logging
import tensorflow_addons as tfa

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def norma_per_epoch(model):
    ww = []
    weights=model.get_weights()
    for w in weights:
        for k in w.ravel():
            ww.append(k)
    return np.linalg.norm(ww)

core_path=r'D:\Loss_Visual\ResNet20_SHCUT_WD_B128\saved_models'

def norma(core_path):
    norma=[]
    for fname in os.listdir(core_path):
        model = load_model(os.path.join(core_path, fname))
        norma.append(norma_per_epoch(model))
    return norma

n=norma(core_path)

fig = plt.figure()
plt.plot(range(len(n)),n)
plt.title('L2 norm per epoch')
fig.savefig(fname='L2_norm_weights.pdf', dpi=300, bbox_inches='tight', format='pdf')

plt.show()


