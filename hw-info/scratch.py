import pycuda.driver as cuda

if __name__ == "__main__":
    cuda.init()


    # Get the device
    device = cuda.Device(0)  # 0 is the first device

    # Get the maximum number of threads per block
    max_threads_per_block = device.get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK)

    print(f"Max threads per block: {max_threads_per_block}")