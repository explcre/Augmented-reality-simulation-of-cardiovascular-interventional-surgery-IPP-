#version 330

uniform sampler2D texUnit;

in vec2 texCoordV;

out vec4 colorOut;

void main() {

	colorOut = texture(texUnit, texCoordV);
}