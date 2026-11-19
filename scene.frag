#version 330
in vec2 uv;
out vec4 out_color;
uniform vec2 center;
uniform float depth, incidence, blur;
uniform sampler2D glyphs;
const vec2 text_min = vec2(.72, .85);
const vec2 text_size = vec2(.105, .026);

void main() {
    float stripes = sin(uv.x * 950.) * sin(uv.y * 740.);
    vec3 background = vec3(.20, .24, .26) + .085 * (1. - blur) * stripes;
    vec2 delta = (uv - center) * vec2(16. / 9., 1.);
    float radius = .125 / depth;
    float circle = 1. - smoothstep(radius - .004, radius, length(delta));
    vec3 blue = vec3(.07, .29, .66) + vec3(.07, .11, .22) * cos(incidence);
    vec3 color = mix(background, blue, circle);
    vec2 label_uv = (uv - text_min) / text_size;
    if (all(greaterThanEqual(label_uv, vec2(0.))) && all(lessThanEqual(label_uv, vec2(1.)))) {
        color = mix(vec3(.04), vec3(.93), texture(glyphs, label_uv).r);
    }
    out_color = vec4(color, 1.);
}
