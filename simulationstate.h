#pragma once

#include "sharedarray.h"
#include "random.h"
#include <tuple>

struct Ball
{
    Vec2f currPos;
    Vec2f prevPos;
    Vec2f acceleration;

    Vec2f new_currPos; // these are for resolving ball-to-ball collisions, not performing verlet integration
    Vec2f new_prevPos;

    float radius = 0.0f;
    float mass = 1.0f;
    uchar4 color;
};

struct SimulationState
{
    // global
    int screenWidth = -1;
    int screenHeight = -1;
    float max_u = 0.0f;
    uchar4* pixels = nullptr;
    SharedArray<Ball> balls;
    float dt = 0.00025f;
    float wallCollisionDampening = 0.99f;
    float ballCollisionDampening = 0.99f;
    float gravity = 50.0f;
};

inline void initSimulation(std::tuple<int, int> screenDim, SimulationState& simState)
{
    simState.screenWidth  = std::get<0>(screenDim);
    simState.screenHeight = std::get<1>(screenDim);
    simState.max_u = (simState.screenWidth / (float)simState.screenHeight);;

    int numBalls = 5;
    for (int i = 0; i < numBalls; i++)
    {
        Ball ball;
        ball.radius = 0.1f;
        ball.currPos = randomVec2f(-0.8f, 0.8f);
        Vec2f initVelocity = randomVec2f(-1.0f, 1.0f);
        normalize(initVelocity);
        ball.prevPos = ball.currPos + (initVelocity / 10000.0f);
        ball.color = randomColor();
        simState.balls.add(ball);
    }
    simState.balls.updateHostToDevice();
}

