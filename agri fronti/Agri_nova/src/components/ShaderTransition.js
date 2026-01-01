import React, { useRef, useImperativeHandle, forwardRef } from 'react';
import { Canvas } from '@react-three/fiber';
import * as THREE from 'three';
import gsap from 'gsap';

const ShaderTransition = forwardRef((props, ref) => {
  const meshRef = useRef();

  useImperativeHandle(ref, () => ({
    out: () => {
      return gsap.to(meshRef.current.material.uniforms.uProgress, {
        value: 1,
        duration: 1,
        ease: 'power2.inOut'
      });
    },
    in: () => {
      return gsap.to(meshRef.current.material.uniforms.uProgress, {
        value: 0,
        duration: 1,
        ease: 'power2.inOut'
      });
    }
  }));

  // Create shader material
  const uniforms = useRef({ uProgress: { value: 0.0 } });

  return (
    <Canvas
      className="fixed inset-0 z-50 pointer-events-none"
      gl={{ alpha: true }}
      camera={{ position: [0, 0, 1] }}
    >
      <mesh ref={meshRef}>
        <planeGeometry args={[2, 2]} />
        <shaderMaterial
          uniforms={uniforms.current}
          vertexShader={
`varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position, 1.0);
}`
          }
          fragmentShader={
`uniform float uProgress;
varying vec2 vUv;
void main() {
  float dissolve = smoothstep(0.0, 1.0, uProgress + (vUv.y - 0.5));
  gl_FragColor = vec4(vec3(dissolve), 1.0);
}`
          }
          transparent
          depthTest={false}
        />
      </mesh>
    </Canvas>
  );
});

export default ShaderTransition;
