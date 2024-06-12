# import matplotlib.pyplot as plt
# import numpy as np

# # plt.ion()
# for i in range(50):
#     y = np.random.random([10,1])
#     plt.plot(y)
#     plt.draw()
#     plt.pause(1)
#     plt.clf()

import matplotlib.pyplot as plt
import numpy as np 

while True:
    #create a bunch of random numbers
    random_array = np.random.randint(0,50, size=(8,8))
    #print the array, just so I know you're not broken
    print(random_array)

    #clear the image because we didn't close it
    plt.clf()

    #show the image
#    plt.figure(figsize=(5, 5))
    plt.imshow(random_array, cmap='hot', interpolation='nearest')
    plt.colorbar()
    print("Pausing...")
    plt.pause(5)

   #uncomment this line and comment the line with plt.clf()
#    plt.close()