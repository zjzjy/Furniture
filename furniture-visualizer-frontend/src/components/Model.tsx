import { useGLTF, Loader } from '@react-three/drei';
import { Suspense, useEffect } from 'react';
import * as THREE from 'three'; // 导入 THREE 以便使用 THREE.Material

interface ModelProps {
  url: string;
}

function Model({ url }: ModelProps) {
  const { scene } = useGLTF(url);

  useEffect(() => {
    if (scene) {
      scene.traverse((child) => {
        // 类型守卫，确保 child 是 Mesh 类型
        if ((child as THREE.Mesh).isMesh) {
          const mesh = child as THREE.Mesh;
          const material = mesh.material as THREE.Material | THREE.Material[];

          if (material) {
            if (Array.isArray(material)) {
              material.forEach(mat => {
                mat.transparent = true;
                // mat.alphaTest = 0.5; // 根据需要取消注释和调整
                // mat.depthWrite = true; // 对于某些透明效果可能需要
                mat.needsUpdate = true;
              });
            } else {
              // 如果是单个材质
              material.transparent = true;
              // material.alphaTest = 0.5; // 根据需要取消注释和调整
              // material.depthWrite = true;
              material.needsUpdate = true;
            }
          }
        }
      });
    }
  }, [scene]);

  // 你可以在这里调整模型的初始缩放、位置、旋转等
  // 例如: scene.scale.set(0.5, 0.5, 0.5);
  //       scene.position.set(0, -1, 0);
  return <primitive object={scene} />;
}

interface ModelViewerProps {
  modelUrl: string;
}

export function ModelViewer({ modelUrl }: ModelViewerProps) {
  if (!modelUrl) return null; // 如果没有模型URL，则不渲染任何内容

  return (
    <Suspense fallback={<Loader />}>
      <Model url={modelUrl} />
    </Suspense>
  );
} 