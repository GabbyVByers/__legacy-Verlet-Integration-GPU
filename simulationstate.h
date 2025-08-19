#pragma once

#include "sharedarray.h"
#include "random.h"
#include <tuple>

struct Ball
{
    float radius = 0.0f;
    uchar4 color;
    Vec2f position;
    Vec2f velocity;
    Vec2f acceleration;
};

struct SimulationState
{
    int screenWidth = -1;
    int screenHeight = -1;
    uchar4* pixels = nullptr;
    SharedArray<Ball> balls;
    float dt = 0.0001f;
};

inline void initSimulation(std::tuple<int, int> screenDim, SimulationState& simState)
{
    simState.screenWidth  = std::get<0>(screenDim);
    simState.screenHeight = std::get<1>(screenDim);

    int numBalls = 100;
    for (int i = 0; i < numBalls; i++)
    {
        Ball ball;
        ball.radius = randomFloat(0.01f, 0.03f);
        ball.position = randomVec2f(-1.0f, 1.0f);
        ball.color = randomColor();
        simState.balls.add(ball);
    }
    simState.balls.updateHostToDevice();

}

