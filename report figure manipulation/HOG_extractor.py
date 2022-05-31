from skimage.feature import hog
from skimage.io import imread, imshow
from skimage import exposure
import matplotlib.pyplot as plt

#read image
img = imread('car.jpg')
# imshow(img)
# plt.show()

#create hog features 
fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True, channel_axis=-1)

# plot

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5)) 
# remove x,y ticks
ax1.axis('off')
ax2.axis('off')
ax1.imshow(img) 
ax1.set_title('Εικόνα Εισόδου') 

# rescale histogram for better display 
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10)) 

ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray) 
ax2.set_title('Ιστόγραμμα Προσανατολισμένων Κλίσεων')
plt.savefig('hog_gr.pdf', format='pdf', dpi=1200)

# Code based on https://www.analyticsvidhya.com/blog/2019/09/feature-engineering-images-introduction-hog-feature-descriptor/ 