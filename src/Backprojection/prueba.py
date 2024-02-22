import imageio

# The following line only needs to run once for a user
# to download the necessary binaries to read HDR.
imageio.plugins.freeimage.download()
img = imageio.imread(hdr_path, format='HDR-FI')

#img es un array de numpy con los valores de la imagen en float32
print(img)
