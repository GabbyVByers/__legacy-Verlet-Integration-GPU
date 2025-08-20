
#include "opengl.h"

__global__ void physicsKernel(SimulationState simState)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= simState.balls.size) return;

    SharedArray<Ball>& balls = simState.balls;
    Ball& ball = balls.devPtr[index];

    float dt = simState.dt;
    Vec2f new_position = ball.position + (ball.velocity * dt) + (ball.acceleration * (dt * dt * 0.5f));
    Vec2f new_acceleration = Vec2f{ 0.0f, -simState.gravity };
    Vec2f new_velocity = ball.velocity + ((ball.acceleration + new_acceleration) * (dt * 0.5f));

    ball.position = new_position;
    ball.velocity = new_velocity;
    ball.acceleration = new_acceleration;
}

__global__ void ballCollisionKernel(SimulationState simState)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= simState.balls.size) return;

    SharedArray<Ball>& balls = simState.balls;
    Ball& ball = balls.devPtr[index];
    ball.new_position = ball.position;
    ball.new_velocity = ball.velocity;
    
    for (int i = 0; i < balls.size; i++)
    {
        if (i == index) continue;
        Ball& otherBall = balls.devPtr[i];
        
        float distance = length(ball.position - otherBall.position);
        float radiuses = ball.radius + otherBall.radius;
        if (distance > radiuses) continue;

        // elastic collision
        float M1 = ball.mass;
        float M2 = otherBall.mass;
        Vec2f V1 = ball.velocity;
        Vec2f V2 = otherBall.velocity;
        Vec2f P1 = ball.position;
        Vec2f P2 = otherBall.position;
        ball.new_velocity = V1 - (P1 - P2) * (((2.0f * M2) / (M1 + M2)) * (dot(V1 - V2, P1 - P2) / lengthSquared(P1 - P2)));
        ball.new_velocity = ball.new_velocity * simState.ballCollisionDampening;

        // force apart
        float overlap = (radiuses - distance);
        Vec2f direction = ball.position - otherBall.position;
        normalize(direction);
        Vec2f correction = direction * overlap;
        ball.new_position = ball.position + correction;
    }
}

__global__ void confirmCollisionsKernel(SimulationState simState)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= simState.balls.size) return;

    SharedArray<Ball>& balls = simState.balls;
    Ball& ball = balls.devPtr[index];

    ball.position = ball.new_position;
    ball.velocity = ball.new_velocity;
}

__global__ void wallCollisionResolutionKernel(SimulationState simState)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= simState.balls.size) return;

    SharedArray<Ball>& balls = simState.balls;
    Ball& ball = balls.devPtr[index];

    if (ball.position.y - ball.radius < -1.0f) // floor
    {
        ball.position.y = -1.0f + ball.radius;
        ball.velocity.y = -1.0f * ball.velocity.y;
        ball.velocity = ball.velocity * simState.wallCollisionDampening;
    }

    if (ball.position.y + ball.radius > 1.0f) // ceiling
    {
        ball.position.y = 1.0f - ball.radius;
        ball.velocity.y = -1.0f * ball.velocity.y;
        ball.velocity = ball.velocity * simState.wallCollisionDampening;
    }

    if (ball.position.x + ball.radius > simState.max_u) // right wall
    {
        ball.position.x = simState.max_u - ball.radius;
        ball.velocity.x = -1.0f * ball.velocity.x;
        ball.velocity = ball.velocity * simState.wallCollisionDampening;
    }

    if (ball.position.x - ball.radius < -simState.max_u) // left wall
    {
        ball.position.x = -simState.max_u + ball.radius;
        ball.velocity.x = -1.0f * ball.velocity.x;
        ball.velocity = ball.velocity * simState.wallCollisionDampening;
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

    for (int i = 0; i < 16; i++)
    {
        ballCollisionKernel <<<BALLS_blocksPerGrid, BALLS_threadsPerBlock>>> (simState);
        cudaDeviceSynchronize();
        confirmCollisionsKernel <<<BALLS_blocksPerGrid, BALLS_threadsPerBlock>>> (simState);
        cudaDeviceSynchronize();
        wallCollisionResolutionKernel <<<BALLS_blocksPerGrid, BALLS_threadsPerBlock >>> (simState);
        cudaDeviceSynchronize();
    }

    physicsKernel << <BALLS_blocksPerGrid, BALLS_threadsPerBlock >> > (simState);
    cudaDeviceSynchronize();

    renderKernel <<<WINDOW_blocksPerGrid, WINDOW_threadsPerBlock >>> (simState);
    cudaDeviceSynchronize();
}

