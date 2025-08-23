
#include "opengl.h"
#define KERNEL_DIM(x, y) <<<x, y>>>

__device__ Vec2f getVelocity(Ball& ball, float dt)
{
    Vec2f newPos = (ball.prevPos * -1.0f) + (ball.currPos * 2.0f) + (ball.acceleration * dt * dt);
    Vec2f velocity = (newPos - ball.prevPos) / (2.0f * dt);
    return velocity;
}

__device__ void setVelocity(Ball& ball, Vec2f velocity, float dt)
{
    ball.prevPos = ball.currPos - (velocity * dt) + (ball.acceleration * (0.5f * dt * dt));
}

__global__ void wallCollisionKernel(SimulationState simState)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= simState.balls.size) return;

    SharedArray<Ball>& balls = simState.balls;
    Ball& ball = balls.devPtr[index];

    float dt = simState.dt;

    if (ball.currPos.y - ball.radius < -1.0f) // floor
    {
        Vec2f velocity = getVelocity(ball, dt);
        velocity.y = fabs(velocity.y);
        ball.currPos.y = -1.0f + ball.radius;
        setVelocity(ball, velocity, dt);
    }

    if (ball.currPos.y + ball.radius > 1.0f) // ceiling
    {
        Vec2f velocity = getVelocity(ball, dt);
        velocity.y = -1.0f * fabs(velocity.y);
        ball.currPos.y = 1.0f - ball.radius;
        setVelocity(ball, velocity, dt);
    }

    if (ball.currPos.x - ball.radius < (-1.0f * simState.max_u)) // left
    {
        Vec2f velocity = getVelocity(ball, dt);
        velocity.x = fabs(velocity.x);
        ball.currPos.x = (-1.0f * simState.max_u) + ball.radius;
        setVelocity(ball, velocity, dt);
    }

    if (ball.currPos.x + ball.radius > simState.max_u) // left
    {
        Vec2f velocity = getVelocity(ball, dt);
        velocity.x = -1.0f * fabs(velocity.x);
        ball.currPos.x = simState.max_u - ball.radius;
        setVelocity(ball, velocity, dt);
    }
}

__global__ void ellasticCollisionKernel(SimulationState simState)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= simState.balls.size) return;

    SharedArray<Ball>& balls = simState.balls;
    Ball& ball = balls.devPtr[index];

    int closestOverlappingOtherBallIndex = -1;
    float closestDistance = FLT_MAX;
    for (int i = 0; i < balls.size; i++)
    {
        if (i == index) continue;
        Ball& otherBall = balls.devPtr[index];

        Vec2f difference = ball.currPos - otherBall.currPos;
        float distance = length(difference);
        float radiuses = ball.radius + otherBall.radius;
        if (distance < radiuses)
        {
            if (distance < closestDistance)
            {
                closestDistance = distance;
                closestOverlappingOtherBallIndex = index;
            }
        }
    }

    ball.bounceVelocity = Vec2f{ 0.0f, 0.0f };
    if (closestOverlappingOtherBallIndex == -1) return;
    Ball& otherBall = balls.devPtr[closestOverlappingOtherBallIndex];
    float dt = simState.dt;
    Vec2f V1 = getVelocity(ball, dt);
    Vec2f V2 = getVelocity(otherBall, dt);
    float M1 = ball.mass;
    float M2 = otherBall.mass;
    Vec2f P1 = ball.currPos;
    Vec2f P2 = otherBall.currPos;
    ball.bounceVelocity = V1 - ((P1 - P2) * ((2.0f * M2) / (M1 + M2)) * (dot(V1 - V2, P1 - P2) / lengthSquared(P1 - P2)));
}

__global__ void setBounceVelocitiesKernel(SimulationState simState)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= simState.balls.size) return;

    SharedArray<Ball>& balls = simState.balls;
    Ball& ball = balls.devPtr[index];
    setVelocity(ball, ball.bounceVelocity, simState.dt);
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
    Vec2f pixelPos = Vec2f{ u,v };

    SharedArray<Ball>& balls = simState.balls;
    for (int i = 0; i < balls.size; i++)
    {
        Ball& ball = balls.devPtr[i];
        Vec2f relBallPos = ball.currPos - pixelPos;
        if (length(relBallPos) < ball.radius)
        {
            simState.pixels[index] = ball.color;
            return;
        }
    }

    simState.pixels[index] = make_uchar4(0, 0, 0, 255);
    return;
}

__global__ void stepPhysicsKernel(SimulationState simState)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= simState.balls.size) return;

    SharedArray<Ball>& balls = simState.balls;
    Ball& ball = balls.devPtr[index];

    Vec2f newPos = (ball.prevPos * -1.0f) + (ball.currPos * 2.0f) + (ball.acceleration * simState.dt * simState.dt);
    ball.prevPos = ball.currPos;
    ball.currPos = newPos;
}

void InteropOpenGL::executeKernels(SimulationState& simState)
{
    simState.pixels = nullptr;
    size_t size = 0;
    cudaGraphicsMapResources(1, &cudaPBO, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&simState.pixels, &size, cudaPBO);

    int BALLS_threadsPerBlock = 256;
    int BALLS_blocksPerGrid = (simState.balls.size + BALLS_threadsPerBlock - 1) / BALLS_threadsPerBlock;

    // determine acceleration kernel (not implemented because currently static)
    wallCollisionKernel       KERNEL_DIM(BALLS_blocksPerGrid, BALLS_threadsPerBlock)(simState); cudaDeviceSynchronize();
    ellasticCollisionKernel   KERNEL_DIM(BALLS_blocksPerGrid, BALLS_threadsPerBlock)(simState); cudaDeviceSynchronize();
    setBounceVelocitiesKernel KERNEL_DIM(BALLS_blocksPerGrid, BALLS_threadsPerBlock)(simState); cudaDeviceSynchronize();
    renderKernel              KERNEL_DIM(PIXLS_blocksPerGrid, PIXLS_threadsPerBlock)(simState); cudaDeviceSynchronize();
    stepPhysicsKernel         KERNEL_DIM(BALLS_blocksPerGrid, BALLS_threadsPerBlock)(simState); cudaDeviceSynchronize();
}

