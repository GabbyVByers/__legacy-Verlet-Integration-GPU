
#include "opengl.h"

__global__ void physicsKernel(SimulationState simState)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= simState.balls.size) return;

    SharedArray<Ball>& balls = simState.balls;
    Ball& ball = balls.devPtr[index];

    float dt = simState.dt;
    Vec2f new_position = ball.position + (ball.velocity * dt) + (ball.acceleration * (dt * dt * 0.5f));
    Vec2f new_acceleration = Vec2f{ 0.0f, -1000.0f };
    Vec2f new_velocity = ball.velocity + ((ball.acceleration + new_acceleration) * (dt * 0.5f));

    ball.position = new_position;
    ball.velocity = new_velocity;
    ball.acceleration = new_acceleration;
}

__global__ void collisionResolutionKernel(SimulationState simState)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= simState.balls.size) return;

    SharedArray<Ball>& balls = simState.balls;
    Ball& ball = balls.devPtr[index];
    
    for (int i = 0; i < balls.size; i++)
    {
        if (i == index) continue;
        Ball& otherBall = balls.devPtr[i];
        
        float distance = length(ball.position - otherBall.position);
        float radiuses = ball.radius + otherBall.radius;
        if (distance > radiuses) continue;

        // elastic bounce
        float massRatio = (2.0f * otherBall.mass) / (ball.mass + otherBall.mass);
        float dotProduct = dot(ball.velocity - otherBall.velocity, ball.position - otherBall.position);
        Vec2f relPosVector = ball.position - otherBall.position;
        float lengthSq = lengthSquared(relPosVector);
        Vec2f new_velocity = ball.velocity - (relPosVector * (massRatio * (dotProduct / lengthSq)));
        ball.velocity = new_velocity;
    }

    for (int i = 0; i < balls.size; i++)
    {
        if (i == index) continue;
        Ball& otherBall = balls.devPtr[i];

        float distance = length(ball.position - otherBall.position);
        float radiuses = ball.radius + otherBall.radius;
        if (distance > radiuses) continue;

        // force apart
        float halfOverlap = (radiuses - distance) * 0.5f;
        Vec2f direction = ball.position - otherBall.position;
        normalize(direction);
        Vec2f correction = direction * halfOverlap;
        ball.position = ball.position + correction;
    }

    if (ball.position.y - ball.radius < -1.0f) // floor
    {
        ball.position.y = -1.0f + ball.radius;
        ball.velocity.y = -ball.velocity.y;
    }

    if (ball.position.y + ball.radius > 1.0f) // ceiling
    {
        ball.position.y = 1.0f - ball.radius;
        ball.velocity.y = -ball.velocity.y;
    }

    if (ball.position.x + ball.radius > simState.max_u) // right wall
    {
        ball.position.x = simState.max_u - ball.radius;
        ball.velocity.x = -ball.velocity.x;
    }

    if (ball.position.x - ball.radius < -simState.max_u) // left wall
    {
        ball.position.x = -simState.max_u + ball.radius;
        ball.velocity.x = -ball.velocity.x;
    }
}

__global__ void renderKernel(SimulationState simState)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = -1;
    if ((x < simState.screenWidth) && (y < simState.screenHeight))
        index = y * simState.screenWidth + x;
    else
        return;
    
    float u = ((x / (float)simState.screenWidth) * 2.0f - 1.0f) * (simState.screenWidth / (float)simState.screenHeight);
    float v = (y / (float)simState.screenHeight) * 2.0f - 1.0f;

    SharedArray<Ball>& balls = simState.balls;
    for (int i = 0; i < balls.size; i++)
    {
        Ball& ball = balls.devPtr[i];
        Vec2f relBallPos = ball.position - Vec2f{ u,v };
        if (length(relBallPos) < ball.radius)
        {
            simState.pixels[index] = ball.color;
            return;
        }
    }

    simState.pixels[index] = make_uchar4(0, 0, 0, 255);
    return;
}

void InteropOpenGL::executePixelKernel(SimulationState& simState)
{
    simState.pixels = nullptr;
    size_t size = 0;
    cudaGraphicsMapResources(1, &cudaPBO, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&simState.pixels, &size, cudaPBO);

    int BALLS_threadsPerBlock = 256;
    int BALLS_blocksPerGrid = (simState.balls.size + BALLS_threadsPerBlock - 1) / BALLS_threadsPerBlock;
    
    physicsKernel <<<BALLS_blocksPerGrid, BALLS_threadsPerBlock>>> (simState);
    cudaDeviceSynchronize();

    collisionResolutionKernel << <BALLS_blocksPerGrid, BALLS_threadsPerBlock >> > (simState);
    cudaDeviceSynchronize();

    renderKernel <<<WINDOW_blocksPerGrid, WINDOW_threadsPerBlock >>> (simState);
    cudaDeviceSynchronize();
}

