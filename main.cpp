
#include "opengl.h"

int main()
{
    InteropOpenGL OpenGL(1200, 800, "Cuda OpenGL Interop", false);

    while (OpenGL.isAlive())
    {
        OpenGL.executePixelKernel();
        OpenGL.renderFullScreenQuad();
        OpenGL.renderImGui();
        OpenGL.swapBuffers();
    }

    return 0;
}

