#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

layout(binding = 0) uniform UBOViewProjection {
    mat4 projection;
    mat4 view;
} uboViewProjection;

layout(binding = 1) uniform UBOModel {
    mat4 model;
} uboModel;

layout(location = 0) out vec3 fragColor;

void main() {
    gl_Position = uboViewProjection.projection * uboViewProjection.view * uboModel.model * vec4(position, 1.0);
    fragColor = color;
}