#pragma once

struct Vec2f
{
    float x = 0.0f;
    float y = 0.0f;

    __host__ __device__ Vec2f operator - (Vec2f& other)
    {
        return
        {
            x - other.x,
            y - other.y
        };
    }
};

__host__ __device__ inline float length(Vec2f& vec)
{
    return sqrt((vec.x * vec.x) + (vec.y * vec.y));
}