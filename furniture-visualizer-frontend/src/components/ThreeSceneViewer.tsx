import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Environment } from '@react-three/drei';
import { ModelViewer } from './Model'; // 导入我们创建的 ModelViewer

interface ThreeSceneViewerProps {
  modelPath: string | null; // 注意这是从API获取的相对路径，如 /static_glb_files/model.glb
  backendBaseUrl: string;  // 后端基础URL，如 http://localhost:8000
}

export function ThreeSceneViewer({ modelPath, backendBaseUrl }: ThreeSceneViewerProps) {
  if (!modelPath) {
    // 如果没有模型路径，可以显示一些提示信息或保持空白
    return <div style={{ width: '100%', height: '400px', display:'flex', justifyContent:'center', alignItems:'center', border:'1px dashed gray' }}>请上传图片以显示3D模型</div>;
  }

  // 拼接完整的模型URL
  const fullModelUrl = `${backendBaseUrl}${modelPath}`;

  return (
    <div style={{ width: '100%', height: '500px', border: '1px solid black' }}>
      <Canvas>
        {/* 环境光和方向光 */}
        <ambientLight intensity={Math.PI / 2} />
        <directionalLight position={[10, 10, 5]} intensity={1} />
        <directionalLight position={[-10, -10, -5]} intensity={0.5} />
        
        {/* 使用Drei的Environment来提供一些基础环境照明和背景 (可选) */}
        <Environment preset="sunset" background blur={0.5} />
        
        {/* 相机设置 */}
        <PerspectiveCamera makeDefault position={[0, 2, 5]} fov={50} />
        
        {/* 轨道控制器，用于鼠标交互 */}
        <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
        
        {/* 加载和显示模型 */}
        <ModelViewer modelUrl={fullModelUrl} />
        
        {/* (可选MVP+) 一个简单的地面平面 */}
        {/* 
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -1.5, 0]}>
          <planeGeometry args={[10, 10]} />
          <meshStandardMaterial color="gray" />
        </mesh>
        */}
      </Canvas>
    </div>
  );
} 