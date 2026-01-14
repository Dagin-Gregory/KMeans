from typing import TypeAlias,Any
import time as time
import KMeans
import numpy as np
#from sklearn.decomposition import PCA

#from scipy.cluster.vq import kmeans, vq, whiten
import numpy as np
import matplotlib.pyplot as plt

DataValue: TypeAlias = np.ubyte | np.short | np.int32 | np.float32 | np.double  # Python 3.10+
PRINT_TIMETAKEN = False
NORMALIZE = True


def readUbyteFile(file_path:str) -> tuple[DataValue, list[int], np.ndarray]:
    with open(file_path, mode='rb') as f:
        magic_number       = f.read(4)
        data_type_code:int = magic_number[2]
        num_dimensions:int = magic_number[3]

        bytes_per_value:int = 1
        data_type:DataValue = np.ubyte
        match data_type_code:
            case 8:
                bytes_per_value = 1
                data_type = np.ubyte
            case 9:
                bytes_per_value = 1
                data_type = np.byte
            case 11:
                bytes_per_value = 2
                data_type = np.short
            case 12:
                bytes_per_value = 4
                data_type = np.int32
            case 13:
                bytes_per_value = 4
                data_type = np.float32
            case 14:
                bytes_per_value = 8
                data_type = np.double
            case _:
                raise Exception('Invalid data type code, file type shouldnt be bytes.')
                bytes_per_value = 1
                data_type = np.byte

        num_items = int.from_bytes(f.read(4), 'big')
        items_per_elem = 1
        channel_dimensions:list[int] = []
        for _ in range(num_dimensions-1):
            curr_dim = int.from_bytes(f.read(4), 'big')
            items_per_elem *= curr_dim
            channel_dimensions.append(curr_dim)

        data = np.empty((num_items*items_per_elem), dtype=data_type)
        rel_elements = 10000
        time_start = time.time()
        total_time = 0.0

        chunk_size = 500
        for elem in range(0, num_items, chunk_size):
            file_return = f.read(chunk_size*items_per_elem*bytes_per_value)
            element_array = np.frombuffer(file_return, dtype=data_type)
            data_idx = elem*items_per_elem
            data[data_idx:data_idx + items_per_elem*chunk_size] = element_array
            if (PRINT_TIMETAKEN and (elem % rel_elements == 0 or elem == num_items-1) and elem != 0):
                elapsed = time.time()-time_start
                total_time += elapsed
                time_start = time.time()
                print(f"Time to proc {rel_elements}: {elapsed*1000:.2f}ms")

        if (PRINT_TIMETAKEN):
            print(f"Total time taken: {total_time*1000:.2f}ms")
        
        if (items_per_elem > 1):
            data = np.reshape(data, (num_items, items_per_elem))
        else:
            data = np.reshape(data, (num_items))
        return (data_type, channel_dimensions, data)

def normalize(vector:np.ndarray) -> np.ndarray:
    #normalized_vector = vector.astype(np.float32) / 255.0
    #return normalized_vector
    vector = vector.astype(np.float32) / 255.0
    return vector / (np.linalg.norm(vector) + 1e-8)


def main():
    train_images_pth = 'MNIST/raw/train-images-idx3-uByte'
    train_labels_pth = 'MNIST/raw/train-labels-idx1-ubyte'
    test_images_pth  = 'MNIST/raw/t10k-images-idx3-ubyte'
    test_labels_pth  = 'MNIST/raw/t10k-labels-idx1-ubyte'
    data_type_train_imgs,   ch_dimensions_train_imgs,   data_train_imgs  =  readUbyteFile(train_images_pth)
    data_type_train_labels, ch_dimensions_train_labels, data_train_labels=  readUbyteFile(train_labels_pth)
    data_type_test_imgs,   ch_dimensions_test_imgs,   data_test_imgs     =  readUbyteFile(test_images_pth)
    data_type_test_labels, ch_dimensions_test_labels, data_test_labels   =  readUbyteFile(test_labels_pth)

    """
    x:np.ndarray = data_train_imgs[0]  # shape (784,)

    # Reshape into 28x28 image
    image = x.reshape(28, 28)

    # Plot
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()
    """

    k = 150
    points = 1000
    vector_length = 1
    for dimension in ch_dimensions_train_imgs:
        vector_length *= dimension

    if (NORMALIZE):
        data_train_imgs = normalize(data_train_imgs)
        data_test_imgs = normalize(data_test_imgs)
    
    cluster = KMeans.KMeans(k, debug=True)
    cluster.train(data_train_imgs[:points], data_train_labels[:points], epochs=4)
    #accuracy, predictions = cluster.test(data_test_imgs[:points], data_test_labels[:points])
    #print(accuracy*100)
    if (points >= data_test_imgs.shape[0]):
        accuracy = cluster.test(data_test_imgs, data_test_labels)
    accuracy = cluster.test(data_test_imgs[:points], data_test_labels[:points])
    print(accuracy)


    #return_arr, return_float = kmeans(data_train_imgs[:points], k, iter=10)

    #print(predictions)
    #cluster.printLabels()

    
if __name__ == '__main__':
    main()
