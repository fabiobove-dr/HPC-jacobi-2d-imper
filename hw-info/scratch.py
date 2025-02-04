import pycuda.driver as cuda

if __name__ == "__main__":
    # Initialize CUDA and get the device
    cuda.init()
    device = cuda.Device(0)  # 0 is the first device

    # Maximum number of threads per block
    max_threads_per_block = device.get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK)
    print(f"Max threads per block: {max_threads_per_block}")
    attributes = device.get_attributes()

    print(attributes)  # Already a dictionary
    # Maximum number of registers per block
    max_registers_per_block = device.get_attribute(cuda.device_attribute.MAX_REGISTERS_PER_BLOCK)
    print(f"Max registers per block: {max_registers_per_block}")

