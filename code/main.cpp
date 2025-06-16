/*
Author: Shreya Chivilkar
Class: ECE6122 (A)
Last Date Modified: 07th October 2024
Description: Lab 2 - Game of Life Code -> This file "main.cpp" contains code that is used to generate the the grid, check if the cell will be alive or dead based on neighbors, replace the grid with the new grid.
I have developed three different methods for Sequential processing, Parallel Processing using std::threads and Open MP.
Then calculate the time required for 100 generations for each of this type based on user input.
*/

#include <vector>
#include <random>
#include <chrono>
#include <SFML/Graphics.hpp>
#include <iostream>
#include <thread>
#include <omp.h>

using namespace std;
using namespace std::chrono;

int getIndex(int x, int y, int width) {
    return x + y * width;
}

//This function would check if the neighbors of a cell lie within the boundary condition and update the count of alive neighbors
int countLiveNeighbors(int x, int y, const std::vector<int>& grid, int width, int height) {
    int liveCount = 0;

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) continue;

            int nx = x + i;
            int ny = y + j;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                
                liveCount += grid[nx+ny*width];
            }
        }
    }
    return liveCount;
}

// This function is used for calculating the values of the new grid sequentially using nested for loops and traversing over the entire grid
void processSequential(std::vector<int>& grid, std::vector<int>& newGrid, int width, int height) {
    std::fill(newGrid.begin(), newGrid.end(), 0);

    for (int y = 0; y < height; y++) {
        int baseIndex = y *width;
        for (int x = 0; x < width; x++) {
            int liveNeighbors = countLiveNeighbors(x, y, grid, width, height);
            int index = x + baseIndex;
            newGrid[index] = (grid[index] == 1 && (liveNeighbors == 2 || liveNeighbors == 3)) ||
                             (grid[index] == 0 && liveNeighbors == 3) ? 1 : 0;
        }
    }
    std::swap(grid, newGrid);
}

//This function uses multithreading to update the grid.
//Divide the grid into smaller grids and then assign each thread with smaller grid for parallel processing
void processParallel(std::vector<int>& grid, std::vector<int>& newGrid, int width, int height, int numThreads) {
    std::fill(newGrid.begin(), newGrid.end(), 0);

    auto threadWorker = [&](int startRow, int endRow) {
        // cout<<std::this_thread::get_id()<<endl;
        // std::cout<<"Number of rows in a thread"<<endRow-startRow<<endl;
        for (int y = startRow; y < endRow; y++) {
            int baseIndex = y * width;
            for (int x = 0; x < width; x++) {
                int idx = baseIndex + x;  
                int liveNeighbors = countLiveNeighbors(x, y, grid, width, height);
                
                newGrid[idx] = (grid[idx] == 1 && (liveNeighbors == 2 || liveNeighbors == 3)) ||
                               (grid[idx] == 0 && liveNeighbors == 3) ? 1 : 0;
            }
        }
    };

    std::vector<std::thread> threads;
    int rowsPerThread = height / numThreads; 
    int remainingRows = height % numThreads;
    int currentRow = 0;
    for (int i = 0; i < numThreads; i++) {
        int startRow = currentRow;
        int endRow = startRow + rowsPerThread + (remainingRows-- > 0 ? 1 : 0);
        threads.emplace_back(threadWorker, startRow, endRow);
        currentRow = endRow;
    }

    for (auto& t : threads) {
        t.join();
    }

    std::swap(grid, newGrid);
}

//This function uses OpenMP to update the grid.
//Use collapse (2) to merge two for loops 
void processOpenMP(std::vector<int>& grid, std::vector<int>& newGrid, int width, int height, int numThreads) {
    #pragma omp parallel for collapse(2) num_threads(numThreads)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int index = getIndex(x, y, width);
            int aliveNeighbors = countLiveNeighbors(x, y, grid, width, height);
            newGrid[index] = (grid[index] == 1 && 
                                               (aliveNeighbors == 2 || aliveNeighbors == 3)) || 
                                              (grid[index] == 0 && (aliveNeighbors == 3)) ? 1 : 0;
        }
    }
    std::swap(grid, newGrid);
}

//This function is used to draw the grid/rectangle
void draw(sf::RenderWindow& window, const std::vector<int>& grid, int width, int height, int cellSize) {
    sf::VertexArray quads(sf::Quads);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (grid[getIndex(x, y, width)] == 1) {
                sf::Vertex quad[4];
                quad[0].position = sf::Vector2f(x * cellSize, y * cellSize);
                quad[1].position = sf::Vector2f((x + 1) * cellSize, y * cellSize);
                quad[2].position = sf::Vector2f((x + 1) * cellSize, (y + 1) * cellSize);
                quad[3].position = sf::Vector2f(x * cellSize, (y + 1) * cellSize);

                for (int i = 0; i < 4; i++) {
                    quad[i].color = sf::Color::White;
                }
                quads.append(quad[0]);
                quads.append(quad[1]);
                quads.append(quad[2]);
                quads.append(quad[3]);
            }
        }
    }
    window.draw(quads);
}

//This function is used to initialise values from the grid randomly.
void randomizeGrid(std::vector<int>& grid, int width, int height) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            grid[getIndex(x, y, width)] = dis(gen);
        }
    }
}

int main(int argc, char* argv[]) {
    //Set default values for the parameters

    int numThreads = 8;
    int cellSize = 5;
    int windowWidth = 800;
    int windowHeight = 600;
    std::string processingType = "THRD";
 
     // Handle the input params and validate based on conditions
    for (int i = 1; i < argc; i += 2) {
        std::string arg = argv[i];

        if (arg == "-n") {
            int numThreadIp = std::atoi(argv[i + 1]);
            if (numThreadIp > 2) {
                numThreads = numThreadIp;
            } else {
                std::cout << "Number of threads should be greater than 2, setting to default value" << endl;
            }
        } else if (arg == "-c") {
            int cellSizeIp = std::atoi(argv[i + 1]);
            if (cellSizeIp > 1) {
                cellSize = cellSizeIp;
            }
        } else if (arg == "-x") {
            windowWidth = std::atoi(argv[i + 1]);
        } else if (arg == "-y") {
            windowHeight = std::atoi(argv[i + 1]);
        } else if (arg == "-t") {
            std::string processingTypeIp = argv[i + 1];
            if (processingTypeIp == "SEQ" || processingTypeIp == "THRD" || processingTypeIp == "OMP") {
                processingType = processingTypeIp;
            }
        } else {
            std::cerr << "Error: Unknown argument " << arg << std::endl;
            return 1;
        }
    }

    if (processingType == "SEQ") {
        numThreads = 1;
    }

    //Calculate the grid width and height based on window height, width and cell size.
    int gridWidth = windowWidth / cellSize;
    int gridHeight = windowHeight / cellSize;
    std::vector<int> grid(gridWidth * gridHeight, 0);
    std::vector<int> newGrid(gridWidth * gridHeight, 0);
    
    randomizeGrid(grid, gridWidth, gridHeight);
    sf::RenderWindow window(sf::VideoMode(windowWidth, windowHeight), "Game of Life");

    high_resolution_clock::time_point last_measurement = high_resolution_clock::now();
    long long total_time_span = 0;
    int generationCount = 0;
    
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed || 
                (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Escape)) {
                window.close();
            }
        }

        high_resolution_clock::time_point start = high_resolution_clock::now();
        //processing based on type selected
        if (processingType == "SEQ") {
            processSequential(grid, newGrid, gridWidth, gridHeight);
        } else if (processingType == "THRD") {
            processParallel(grid, newGrid, gridWidth, gridHeight, numThreads);
        } else if (processingType == "OMP") {
            processOpenMP(grid, newGrid, gridWidth, gridHeight, numThreads);
        }

        high_resolution_clock::time_point end = high_resolution_clock::now();
        
        duration<double, std::micro> time_span = duration_cast<duration<double>>(end - start);
        total_time_span += time_span.count();
        generationCount++;
        
        //Calculate the time for 100 iterations
        if (generationCount % 100 == 0) {
            if (processingType == "SEQ") {
                std::cout << "100 generations took " << total_time_span 
                     << " microseconds with single thread."<< std::endl;
            }
            else if (processingType == "THRD") {
                std::cout << "100 generations took " << total_time_span 
                     << " microseconds with "<< numThreads<<" std::threads."<<std::endl;
            }
            else if (processingType == "OMP") {
                std::cout << "100 generations took " << total_time_span 
                     << " microseconds with "<< numThreads<<" OMP threads."<<std::endl;
                }
            total_time_span = 0;
        }

        window.clear();
        draw(window, grid, gridWidth, gridHeight, cellSize);
        window.display();
    }

    return 0;
}