#pragma once

#include "sharedarray.h"
#include "random.h"
#include <tuple>

struct Ball
{
    float radius = 0.0f;
    float mass = 1.0f;
    uchar4 color;
    
    Vec2f new_position;
    Vec2f new_velocity;

    Vec2f position;
    Vec2f velocity;
    Vec2f acceleration;
};

struct SimulationState
{
    // global
    int screenWidth = -1;
    int screenHeight = -1;
    float max_u = 0.0f;
    uchar4* pixels = nullptr;
    SharedArray<Ball> balls;
    float dt = 0.00005f;
    float wallCollisionDampening = 0.99f;
    float ballCollisionDampening = 0.99f;
    float gravity = 1000.0f;
    
    // host only
    float globalControl = 0.005f;
};

inline void initSimulation(std::tuple<int, int> screenDim, SimulationState& simState)
{
    simState.screenWidth  = std::get<0>(screenDim);
    simState.screenHeight = std::get<1>(screenDim);
    simState.max_u = (simState.screenWidth / (float)simState.screenHeight);;

    int numBalls = 100;
    for (int i = 0; i < numBalls; i++)
    {
        Ball ball;
        ball.radius = 0.075f;
        ball.position = randomVec2f(-1.0f, 1.0f);
        ball.velocity = randomVec2f(-1.0f, 1.0f);
        normalize(ball.velocity);
        ball.color = randomColor();
        simState.balls.add(ball);
    }
    simState.balls.updateHostToDevice();
}

