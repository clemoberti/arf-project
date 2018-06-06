import matplotlib.pyplot as plt
import numpy as np

def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def show_sample_usps(data):
    w=10
    h=10
    fig=plt.figure(figsize=(10, 10))
    columns = 7
    rows = 5
    for i in range(1, columns*rows +1):
        img = np.random.randint(10, size=(h,w))
        fig.add_subplot(rows, columns, i)
        show_usps(data[i])
        
def show_inpainting_images(image_processing, orig, result, columns, rows):
    w=20
    h=20
    fig=plt.figure(figsize=(w, h))
    results_n = columns - 1
    N = len(orig) + len(result[0]) * results_n
    ind = list(range(1, N + 1, columns))
    for i in range(len(orig)):
        fig.add_subplot(rows, columns, ind[i])
        image_processing.show_im(orig[i], title="Avec bruit",show=False)
    
    for j in range(results_n):
        ind = list(range(2 + j, N + 1, columns))
        for i in range(len(result[j])):
            fig.add_subplot(rows, columns, ind[i])
            h, step, img = result[j][i]
            title = "Resultat, h = " + str(h) + ", step = " + str(step)
            image_processing.show_im(img, title=title,show=False)
    plt.show()

def show_usps(data):
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")
