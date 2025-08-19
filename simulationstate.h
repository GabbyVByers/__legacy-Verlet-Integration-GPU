#pragma once

#include "sharedarray.h"
#include <tuple>
#include "vec2f.h"

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
};

inline float randomFloat(float min, float max)
{
    return ((rand() / (float)RAND_MAX) * (max - min)) + min;
}

inline Vec2f randomVec2f(float min, float max)
{
    return
    {
        randomFloat(min, max),
        randomFloat(min, max)
    };
}

inline uchar4 randomColor()
{
    unsigned char R = rand() % 255;
    unsigned char G = rand() % 255;
    unsigned char B = rand() % 255;
    return make_uchar4(R, G, B, 255);
}

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

