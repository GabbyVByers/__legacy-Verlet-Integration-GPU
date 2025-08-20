
#include "opengl.h"

void InteropOpenGL::initImGui()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.FontGlobalScale = 2.0f;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
}

void InteropOpenGL::renderImGui(SimulationState& simState)
{
    float keneticEnergy = 0.0f;
    float potentialEnergy = 0.0f;

    simState.balls.updateDeviceToHost();
    for (int i = 0; i < simState.balls.size; i++)
    {
        Ball& ball = simState.balls.hostPtr[i];
        float velocity = length(ball.velocity);
        keneticEnergy += 0.5f * ball.mass * velocity * velocity;
        potentialEnergy += ball.mass * (ball.position.y + 1.0f) * simState.gravity;
    }

    float totalEnergy = keneticEnergy + potentialEnergy;
    
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    
    ImGui::Begin("Debugger");
    ImGui::Text("KENETIC   ENERGY: %f", keneticEnergy);
    ImGui::Text("POTENTIAL ENERGY: %f", potentialEnergy);
    ImGui::Text("TOTAL     ENERGY: %f", totalEnergy);

    if (ImGui::Button("DAMPEN") || (ImGui::IsItemActive() && ImGui::IsMouseDown(0)))
    {
        for (int i = 0; i < simState.balls.size; i++)
        {
            Ball& ball = simState.balls.hostPtr[i];
            ball.velocity = ball.velocity * (1.0f - simState.globalControl);
        }
        simState.balls.updateHostToDevice();
    }

    ImGui::End();
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

