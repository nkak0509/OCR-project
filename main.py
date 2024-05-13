import numpy as np

def main():
    f = open("Data\\t10k-images.idx3-ubyte","rb")
    image_size = 28
    num_images = 5
    f.read(16)
    buf = f.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, image_size, image_size, 1)

if __name__ == "__main__":
    main()