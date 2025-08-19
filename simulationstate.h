#pragma once

#include "sharedarray.h"
#include "random.h"
#include <tuple>

struct Ball
{
    float radius = 0.0f;
    float mass = 1.0f;
    uchar4 color;
    Vec2f position;
    Vec2f velocity;
    Vec2f acceleration;
};

struct SimulationState
{
    int screenWidth = -1;
    int screenHeight = -1;
    float max_u = 0.0f;
    uchar4* pixels = nullptr;
    SharedArray<Ball> balls;
    float dt = 0.0001f;
    float wallCollisionDampening = 1.0f;
    float ballCollisionDampening = 1.0f;
};

inline void initSimulation(std::tuple<int, int> screenDim, SimulationState& simState)
{
    simState.screenWidth  = std::get<0>(screenDim);
    simState.screenHeight = std::get<1>(screenDim);
    simState.max_u = (simState.screenWidth / (float)simState.screenHeight);;

    int numBalls = 20;
    for (int i = 0; i < numBalls; i++)
    {
        Ball ball;
        ball.radius = randomFloat(0.1f, 0.15f);
        ball.position = randomVec2f(-1.0f, 1.0f);
        ball.velocity = randomVec2f(-1.0f, 1.0f);
        normalize(ball.velocity);
        ball.velocity = ball.velocity * 30.0f;
        ball.color = randomColor();
        simState.balls.add(ball);
    }
    simState.balls.updateHostToDevice();
}

