from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import h5py
from pandas_ml import ConfusionMatrix

%matplotlib inline
y_true = y_test.argmax(axis=1)
preds = model.predict(prepare(X_test))
preds = preds.argmax(axis=1)
cm = ConfusionMatrix(y_true, preds)
cm.plot(backend='seaborn', normalized=True)
plt.title('Confusion Matrix Stars prediction')
plt.figure(figsize=(12, 10))