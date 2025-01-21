#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>  // Include ctype.h for character classification functions like isspace

#define MAX_BUFFER_SIZE 256  // Define the maximum buffer size for storing output

// Function to execute a command and capture the output
void execute_command(const char *command, char *output) {
    FILE *fp = popen(command, "r");  // Open the command for reading
    if (fp == NULL) {
        perror("popen failed");
        exit(1);
    }
    fgets(output, MAX_BUFFER_SIZE, fp);  // Capture the output of the command
    pclose(fp);  // Close the command stream
}

// Function to clean and format the string by removing unwanted characters
void clean_string(char *str) {
    // Remove leading spaces
    while (isspace(*str)) {
        str++;
    }

    // Remove trailing spaces
    char *end = str + strlen(str) - 1;
    while (end > str && isspace(*end)) {
        end--;
    }
    *(end + 1) = '\0';  // Null-terminate the string at the last non-space character

    // Remove any non-printable characters (ASCII < 32 or 127)
    for (char *ptr = str; *ptr; ptr++) {
        if (*ptr < 32 || *ptr == 127) {
            *ptr = '\0';  // Replace non-printable characters with null terminator
            break;
        }
    }
}

// Function to generate a JSON formatted data and save it to a file
void generate_json(char *cpu_info, char *cpu_cores, char *cpu_architecture, char *cpu_cache,
                   char *memory_total, char *memory_free, char *memory_used, char *swap_total, char *swap_free,
                   char *disk_info, char *gpu_info, char *gpu_memory_info, char *gpu_utilization_info, char *gpu_temperature_info, const char *filename) {
    FILE *file = fopen(filename, "w");  // Open file for writing JSON
    if (file == NULL) {
        perror("fopen failed");
        exit(1);
    }

    fprintf(file, "{\n");

    // CPU info section
    fprintf(file, "  \"cpu\": {\n");
    fprintf(file, "    \"model\": \"%s\",\n", cpu_info);
    fprintf(file, "    \"cores\": \"%s\",\n", cpu_cores);
    fprintf(file, "    \"architecture\": \"%s\",\n", cpu_architecture);
    fprintf(file, "    \"cache\": \"%s\"\n", cpu_cache);
    fprintf(file, "  },\n");

    // Memory info section
    fprintf(file, "  \"memory\": {\n");
    fprintf(file, "    \"total\": \"%s\",\n", memory_total);
    fprintf(file, "    \"free\": \"%s\",\n", memory_free);
    fprintf(file, "    \"used\": \"%s\"\n", memory_used);
    fprintf(file, "  },\n");

    // Swap info section
    fprintf(file, "  \"swap\": {\n");
    fprintf(file, "    \"total\": \"%s\",\n", swap_total);
    fprintf(file, "    \"free\": \"%s\"\n", swap_free);
    fprintf(file, "  },\n");

    // Disk info section
    fprintf(file, "  \"disk\": {\n");
    fprintf(file, "    \"available\": \"%s\"\n", disk_info);
    fprintf(file, "  },\n");

    // GPU info section
    fprintf(file, "  \"gpu\": {\n");
    fprintf(file, "    \"model\": \"%s\",\n", gpu_info);
    fprintf(file, "    \"memory\": \"%s\",\n", gpu_memory_info);
    fprintf(file, "    \"utilization\": \"%s\",\n", gpu_utilization_info);
    fprintf(file, "    \"temperature\": \"%s\"\n", gpu_temperature_info);
    fprintf(file, "  }\n");

    fprintf(file, "}\n");

    fclose(file);  // Close the file after writing
}

int main() {
    // Declare variables to store hardware specifications
    char cpu_info[MAX_BUFFER_SIZE];
    char cpu_cores[MAX_BUFFER_SIZE];
    char cpu_architecture[MAX_BUFFER_SIZE];
    char cpu_cache[MAX_BUFFER_SIZE];

    char memory_total[MAX_BUFFER_SIZE];
    char memory_free[MAX_BUFFER_SIZE];
    char memory_used[MAX_BUFFER_SIZE];
    char swap_total[MAX_BUFFER_SIZE];
    char swap_free[MAX_BUFFER_SIZE];

    char disk_info[MAX_BUFFER_SIZE];
    char gpu_info[MAX_BUFFER_SIZE];
    char gpu_memory_info[MAX_BUFFER_SIZE];
    char gpu_utilization_info[MAX_BUFFER_SIZE];
    char gpu_temperature_info[MAX_BUFFER_SIZE];

    // Get CPU model info
    execute_command("lscpu | grep 'Model name' | cut -d: -f2 | awk '{$1=$1; print}'", cpu_info);
    clean_string(cpu_info);

    // Get CPU cores (physical and logical)
    execute_command("lscpu | grep 'CPU(s):' | cut -d: -f2| awk '{$1=$1; print}'", cpu_cores);
    clean_string(cpu_cores);

    // Get CPU architecture
    execute_command("lscpu | grep 'Architecture' | cut -d: -f2| awk '{$1=$1; print}'", cpu_architecture);
    clean_string(cpu_architecture);

    // Get CPU cache size
    execute_command("lscpu | grep 'L3 cache' | cut -d: -f2| awk '{$1=$1; print}'", cpu_cache);
    clean_string(cpu_cache);

    // Get total memory info
    execute_command("free -h | grep 'Mem' | awk '{print $2}'", memory_total);
    clean_string(memory_total);

    // Get free memory
    execute_command("free -h | grep 'Mem' | awk '{print $4}'", memory_free);
    clean_string(memory_free);

    // Get used memory
    execute_command("free -h | grep 'Mem' | awk '{print $3}'", memory_used);
    clean_string(memory_used);

    // Get swap memory info
    execute_command("free -h | grep 'Swap' | awk '{print $2}'", swap_total);
    clean_string(swap_total);

    execute_command("free -h | grep 'Swap' | awk '{print $3}'", swap_free);
    clean_string(swap_free);

    // Get available disk space info
    execute_command("df -h | grep '^/dev' | awk '{print $4}'", disk_info);
    clean_string(disk_info);

    // Get GPU info (using nvidia-smi for NVIDIA GPUs)
    execute_command("nvidia-smi --query-gpu=gpu_name --format=csv,noheader", gpu_info);
    if (strlen(gpu_info) == 0) {
        // Fallback if no NVIDIA GPU, check for other GPUs
        execute_command("lspci | grep -i 'vga' | awk -F ': ' '{print $2}'", gpu_info);
    }
    clean_string(gpu_info);

    // Get GPU memory info (total, free, used)
    execute_command("nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv,noheader", gpu_memory_info);
    clean_string(gpu_memory_info);

    // Get GPU utilization
    execute_command("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader", gpu_utilization_info);
    clean_string(gpu_utilization_info);

    // Get GPU temperature
    execute_command("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader", gpu_temperature_info);
    clean_string(gpu_temperature_info);

    // Generate JSON file to save system hardware specifications
    generate_json(cpu_info, cpu_cores, cpu_architecture, cpu_cache,
                  memory_total, memory_free, memory_used, swap_total, swap_free,
                  disk_info, gpu_info, gpu_memory_info, gpu_utilization_info, gpu_temperature_info, "system_hw_specs.json");

    printf("System hardware specifications saved in system_hw_specs.json\n");

    return 0;
}
