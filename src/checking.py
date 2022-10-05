import sys
import tensorflow.keras
import pandas as pd
import sklearn as sk
import scipy as sp
import tensorflow as tf
import platform

print(f'Python platform:{platform.platform()}')
print(f'Tensor Flow Version:{tf.__version__}')
print(f'KerasVersion:{tensorflow.keras.__version__}')
print()
print(f'Python version:{sys.version}')
print(f'Pandas:{pd.__version__}')
print(f'Scikit-Learn:{sk.__version__}')
gpu = len(tf.config.list_physical_devices('GPU')) > 0
print("GPU is", 'available' if gpu else 'Not available')
