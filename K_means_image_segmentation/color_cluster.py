from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

im = Image.open("input.jpg")
np_im = np.array(im)

fig = plt.figure( figsize = (6,6))
plt.imshow(np_im)
plt.show()

height = np.shape(np_im)[0]
width = np.shape(np_im)[1]

for n_clusters in range(2, 100):
    X = np.reshape(np_im, [height*width,3])
    X = X.copy()
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)

    for i in range(n_clusters):
        X[np.argwhere(kmeans.labels_==i),:] = np.mean(X[np.argwhere(kmeans.labels_==i),:],axis=0)

    Y = np.reshape(X, [height,width,3])

    im_y = Image.fromarray(Y)
    im_y.save(f'result/{n_clusters}.jpg')

    print(n_clusters)
    fig = plt.figure( figsize = (6,6))
    plt.imshow(Y)
    plt.show()