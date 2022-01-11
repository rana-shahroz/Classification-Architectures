import matplotlib.pyplot as plt
import numpy as np
    
# De-normalizes the pictures for debugging.
def imageShow(image):
    image = image / 2 + 0.5
    numpyImage = image.numpy()
    plt.imshow(np.transpose(numpyImage, (1, 2, 0)))
    plt.show()
