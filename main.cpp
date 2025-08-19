
#include "opengl.h"
#include "simulationstate.h"

int main()
{
    bool fullScreen = false;
    InteropOpenGL OpenGL(1200, 800, "Cuda OpenGL Interop", fullScreen);
    OpenGL.disableVSYNC();

    SimulationState simState;
    initSimulation(OpenGL.getScreenDim(), simState);

    while (OpenGL.isAlive())
    {
        OpenGL.executePixelKernel(simState);
        OpenGL.renderFullScreenQuad();
        OpenGL.renderImGui();
        OpenGL.swapBuffers();
    }

    return 0;
}

