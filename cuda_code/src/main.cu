/*
Author: Shreya Chivilkar
Last Date Modified: 8th November 2024
Description: CUDA-based John Conway’s Game of Life -> This file "main.cpp" contains code that is used to generate the the grid, check if the cell will be alive or dead based on neighbors, replace the grid with the new grid.
I have developed functions for the three types of memory mentioned - Normal, Pinned, Managed 
*/

#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cuda_runtime.h>

//Defined the default values of parameters
const int DEFAULT_THREADS = 32;
const int DEFAULT_CELL_SIZE = 5;
const int DEFAULT_WIDTH = 800;
const int DEFAULT_HEIGHT = 600;
const int UPDATES_PER_BATCH = 100;

//This is the code for the kernel function - it is used for counting live neighbors and updating cells based on the neighbours
__global__ void updateGridKernel(int* logicGrid, int* newGrid, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < rows * cols) {
        int liveNeighbors = 0;
        int row = x / cols;
        int col = x % cols;

        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                //This will skip the current cell
                if (i == 0 && j == 0) continue;
                int nx = (row + i + rows) % rows;
                int ny = (col + j + cols) % cols;
                liveNeighbors += logicGrid[nx * cols + ny];
            }
        }

        /*Applying the game of life rules:
        Checked the count of live neighbours for the current cell
        If the cell is alive - it will become dead when live neighbors are less than 2 or greater than 3, otherwise it will remain alive
        Similarly, if the cell is dead - it will become alive only if the live neighbours count is 3, otherwise it will remain dead
        */
        int currentState = logicGrid[x];
        if (currentState == 1 && (liveNeighbors < 2 || liveNeighbors > 3)) {
            newGrid[x] = 0;  
        } else if (currentState == 0 && liveNeighbors == 3) {
            newGrid[x] = 1; 
        } else {
            newGrid[x] = currentState;
        }
    }
}

void updateGridGPUNormal(int* logicGrid, int rows, int cols, int numThreads) {
    int gridSize = rows * cols * sizeof(int);
    int *d_logicGrid, *d_newGrid;

    //Allocate the memory 
    cudaMalloc(&d_logicGrid, gridSize);
    cudaMalloc(&d_newGrid, gridSize);

    //Copy the grid values from the CPU to GPU (Host to Device)
    cudaMemcpy(d_logicGrid, logicGrid, gridSize, cudaMemcpyHostToDevice);

    //Assign the dimensions
    dim3 blockDim(numThreads);
    dim3 gridDim((rows*cols + blockDim.x - 1) / blockDim.x);

    //Update the grid
    for (int i = 0; i < UPDATES_PER_BATCH; ++i) {
        updateGridKernel<<<gridDim, blockDim>>>(d_logicGrid, d_newGrid, rows, cols);
        cudaDeviceSynchronize();
        std::swap(d_logicGrid, d_newGrid);
    }
    
    // Copy the Updated grid from GPU to CPU (Device to Host)
    cudaMemcpy(logicGrid, d_logicGrid, gridSize, cudaMemcpyDeviceToHost);
    
    // Free the allocated memory
    cudaFree(d_logicGrid);
    cudaFree(d_newGrid); 
}

void updateGridGPUPinned(int* logicGrid, int rows, int cols, int numThreads) {
    int gridSize = rows * cols * sizeof(int);
    int *d_logicGrid, *d_newGrid, *h_logicGrid;

    //Allocate the memory 
    cudaHostAlloc(&h_logicGrid, gridSize, cudaHostAllocDefault);
    cudaMalloc(&d_logicGrid, gridSize);
    cudaMalloc(&d_newGrid, gridSize);

    //Copy the grid values from cpu to pinned memory of cpu (Host to Host)
    cudaMemcpy(h_logicGrid, logicGrid, gridSize, cudaMemcpyHostToHost);
    //Copy the grid values from pinned memory of cpu to GPU (Host to Device)
    cudaMemcpy(d_logicGrid, h_logicGrid, gridSize, cudaMemcpyHostToDevice);

    //Assign the dimensions
    dim3 blockDim(numThreads);
    dim3 gridDim((rows * cols + blockDim.x - 1) / blockDim.x);

    //Update the grid
    for (int i = 0; i < UPDATES_PER_BATCH; ++i) {
        updateGridKernel<<<gridDim, blockDim>>>(d_logicGrid, d_newGrid, rows, cols);
        cudaDeviceSynchronize();
        std::swap(d_logicGrid, d_newGrid);
    }
    
    // Copy the updated grid values from GPU to CPU (Device to host)
    cudaMemcpy(logicGrid, d_logicGrid, gridSize, cudaMemcpyDeviceToHost);
    
    // Free the allocated memory
    cudaFreeHost(logicGrid);
    cudaFree(d_newGrid);
    cudaFree(d_logicGrid);
}

void updateGridGPUManaged(int* logicGrid, int rows, int cols, int numThreads) {
    int gridSize = rows * cols * sizeof(int);
    int *d_logicGrid, *d_newGrid;

    //Allocate the memory 
    cudaError_t error = cudaMallocManaged(&d_logicGrid, gridSize);
    if(error!=cudaSuccess){
        std::cerr<<"Error for memory allocation: "<<cudaGetErrorString(error)<<std::endl;
        return;
    }
    cudaMemcpy(d_logicGrid, logicGrid, gridSize, cudaMemcpyHostToDevice);

    error = cudaMallocManaged(&d_newGrid, gridSize);
    if(error!=cudaSuccess){
        std::cerr<<"Error for memory allocation: "<<cudaGetErrorString(error)<<std::endl;
        return;
    }
    
    // Set the dimensions
    dim3 blockDim(numThreads);
    dim3 gridDim((rows * cols + blockDim.x - 1) / blockDim.x);
    
    //Update the grid
    for (int i = 0; i < UPDATES_PER_BATCH; ++i) {
        updateGridKernel<<<gridDim, blockDim>>>(d_logicGrid, d_newGrid, rows, cols);
        cudaDeviceSynchronize();
        std::swap(d_logicGrid, d_newGrid);
    }
    
    // Copy the updated grid values from GPU to CPU (Device to host)
    error = cudaMemcpy(logicGrid, d_logicGrid, gridSize, cudaMemcpyDeviceToHost);

    if(error!=cudaSuccess){
        std::cerr<<"Error during memory copy: "<<cudaGetErrorString(error)<<std::endl;
        return;
    }
    
    // Free the allocated memory
    cudaFree(d_logicGrid);
    cudaFree(d_newGrid);
    
}

int main(int argc, char* argv[]) {
    //Set default values for the parameters
    int numThreads = DEFAULT_THREADS;
    int cellSize = DEFAULT_CELL_SIZE;
    int window_width = DEFAULT_WIDTH;
    int window_height = DEFAULT_HEIGHT;
    std::string processingType = "normal";

     // Handle the input params and validate based on conditions
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-n" && i + 1 < argc) {
            numThreads = std::stoi(argv[++i]);
            if (numThreads < 32) {
                std::cerr << "Number of threads should be greater than 32, setting to default value\n";
                return 1;
            }
        } else if (arg == "-c" && i + 1 < argc) {
            cellSize = std::stoi(argv[++i]);
            if (cellSize < 1) {
                std::cerr << "Cell size must be greater than or equal to 1.\n";
                return 1;
            }
        } else if (arg == "-x" && i + 1 < argc) {
            window_width = std::stoi(argv[++i]);
        } else if (arg == "-y" && i + 1 < argc) {
            window_height = std::stoi(argv[++i]);
        } else if (arg == "-t" && i + 1 < argc) {
            processingType = argv[++i];
        }
    }

    int gridWidth = window_width / cellSize;
    int gridHeight = window_height / cellSize;

    // Initialise the logic grid
    std::vector<int> logicGrid(gridHeight * gridWidth, 0);
    for (int i = 0; i < gridHeight * gridWidth; ++i) {
        logicGrid[i] = (std::rand() % 2);
    }

    sf::RenderWindow window(sf::VideoMode(window_width, window_height), "CUDA-based John Conway’s Game of Life");

    
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed || sf::Keyboard::isKeyPressed(sf::Keyboard::Escape)) window.close();
        }
        auto startTime = std::chrono::high_resolution_clock::now();
        //processing based on type selected

       
        if(processingType=="managed"){
            updateGridGPUManaged(logicGrid.data(), gridHeight, gridWidth, numThreads);
        }
        else if(processingType=="pinned"){
            updateGridGPUPinned(logicGrid.data(), gridHeight, gridWidth, numThreads);
        }
        else{
            updateGridGPUNormal(logicGrid.data(), gridHeight, gridWidth, numThreads);
        }
        //Calculate the time for 100 iterations
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);

        std::cout << UPDATES_PER_BATCH << " generations took " << duration.count() << " microsecs with " << numThreads << " threads per block using "
                  << processingType << " memory allocation"<<std::endl;
        window.clear();


        for (int i = 0; i < gridHeight; ++i) {
            for (int j = 0; j < gridWidth; ++j) {
                sf::RectangleShape cell(sf::Vector2f(cellSize, cellSize));
                cell.setPosition(j * cellSize, i * cellSize);
                cell.setFillColor(logicGrid[i * gridWidth + j] ? sf::Color::White : sf::Color::Black);
                window.draw(cell);
            }
        }

        window.display();
    }
    return 0;
}
