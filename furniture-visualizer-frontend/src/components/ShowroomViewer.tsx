import { Canvas, type ThreeEvent } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, useGLTF, TransformControls, Box, Select } from '@react-three/drei';
import { EffectComposer, Outline, Selection } from '@react-three/postprocessing';
import { BlendFunction } from 'postprocessing';
import { Suspense, useEffect, useRef, useCallback } from 'react';
import { useAppContext } from '../contexts/AppContext';
import { type IFurniture } from '../contexts/AppContext'; // 导入家具类型接口
import * as THREE from 'three'; // 导入 THREE

// 临时的单个家具渲染组件 (可以后续拆分到 FurnitureItem.tsx)
function Furniture3DModel({ furniture }: { furniture: IFurniture }) {
  const { scene } = useGLTF(furniture.modelUrl); // 假设 modelUrl 是完整的可访问路径
  const { setSelectedFurniture, selectedFurnitureId } = useAppContext(); // 新的，设置ID和3D对象
  const isSelected = selectedFurnitureId === furniture.id;

  useEffect(() => {
    if (scene) {
      // 将家具ID赋给场景对象的name，以便后续TransformControls可能需要通过name查找 (虽然我们现在直接传递对象引用)
      scene.name = furniture.id; 
      scene.traverse((child) => {
        if ((child as THREE.Mesh).isMesh) {
          const mesh = child as THREE.Mesh;
          mesh.userData = { id: furniture.id }; // 可以附加额外数据
          if (Array.isArray(mesh.material)) {
            mesh.material.forEach((mat: THREE.Material) => {
              console.log('Material type (array):', mat.type, 'Color:', (mat as any).color?.getHexString());
              if((mat as any).map) {
                console.log('Texture map (array item):', (mat as any).map);
              } else {
                console.log('No texture map found on array material item.');
              }
              (mat as THREE.MeshStandardMaterial).transparent = true;
              // (mat as any).color = new THREE.Color(0xffffff); // DEBUG: Force color to white
              mat.needsUpdate = true; // 确保材质更新生效
            });
          } else {
            const material = mesh.material as THREE.MeshStandardMaterial; // 或者更通用的 THREE.Material
            console.log('Material type (single):', material.type, 'Color:', material.color?.getHexString());
            if(material.map) {
              console.log('Texture map (single):', material.map);
            } else {
              console.log('No texture map found on single material.');
            }
            material.transparent = true;
            // material.color = new THREE.Color(0xffffff); // DEBUG: Force color to white
            material.needsUpdate = true; // 确保材质更新生效
          }
        }
      });
    }
  }, [scene, furniture.id]);

  const handleClick = useCallback((event: ThreeEvent<MouseEvent>) => {
    event.stopPropagation();
    // console.log('Clicked on furniture:', furniture.id, scene);
    setSelectedFurniture(furniture.id, scene); // 传递整个scene对象作为选中的3D对象
  }, [furniture, scene, setSelectedFurniture]);

  // Wrap the primitive with <Select> for highlighting
  return (
    // @ts-ignore - Temporarily ignore if linter complains about 'enabled' prop
    <Select enabled={isSelected}> 
      <primitive 
        object={scene} 
        position={furniture.position} 
        rotation={furniture.rotation}
        scale={furniture.scale}
        onClick={handleClick}
      />
    </Select>
  );
}

export function ShowroomViewer() {
  const {
    // showroomModelUrl, // Not used for now as we build a white box room
    furnitureInScene,
    selectedFurnitureId, 
    selectedObject, // 从 context 获取选中的 3D 对象
    setSelectedFurniture, // 新的，用于取消选中等
    updateFurnitureInScene,
  } = useAppContext();

  const orbitControlsRef = useRef<any>(); // 用于引用 OrbitControls
  const transformControlsRef = useRef<any>(); // Ref for TransformControls

  // 根据 selectedFurnitureId 找到选中的家具数据 (IFurniture)
  const selectedFurnitureData = furnitureInScene.find(f => f.id === selectedFurnitureId);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setSelectedFurniture(null, null); // 按 ESC 取消选中
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [setSelectedFurniture]);

  // Effect to sync TransformControls with OrbitControls for dragging
  useEffect(() => {
    if (transformControlsRef.current) {
      const controls = transformControlsRef.current;
      const callback = (event: any) => { // Changed to any to avoid problematic casting
        if (orbitControlsRef.current && typeof event.value === 'boolean') {
          orbitControlsRef.current.enabled = !event.value;
        }
      };
      controls.addEventListener('dragging-changed', callback); 
      return () => controls.removeEventListener('dragging-changed', callback);
    }
  }, [selectedObject]); // Re-run if selectedObject changes, so event listener is on correct TransformControls

  const handleTransformEnd = useCallback(() => {
    if (selectedObject && selectedFurnitureData && transformControlsRef.current) {
      // It's possible that selectedObject is the one from context,
      // but transformControlsRef.current.object is the one that TransformControls actually manipulated.
      // Prefer transformControlsRef.current.object if available and has new data.
      const manipulatedObject = transformControlsRef.current.object as THREE.Object3D || selectedObject;
      
      const updatedFurniture: IFurniture = {
        ...selectedFurnitureData,
        position: [manipulatedObject.position.x, manipulatedObject.position.y, manipulatedObject.position.z],
        rotation: [manipulatedObject.rotation.x, manipulatedObject.rotation.y, manipulatedObject.rotation.z],
        scale: [manipulatedObject.scale.x, manipulatedObject.scale.y, manipulatedObject.scale.z],
      };
      updateFurnitureInScene(updatedFurniture);
    }
  }, [selectedObject, selectedFurnitureData, updateFurnitureInScene]);

  return (
    <div style={{ width: '100%', height: 'calc(100vh - 250px)', border: '1px solid lightblue' }}>
      <Canvas
        // scene={{ background: new THREE.Color(0xffffff) }} // Set background color here
        // or use style on canvas element: style={{ background: 'white' }}
        // For R3F, setting scene background or a clear color on gl is better
        gl={{ antialias: true, alpha: false }} // Ensure alpha is false for solid background
        onCreated={({ gl }) => {
          gl.setClearColor(0xffffff, 1); // Set clear color to white
        }}
        onPointerMissed={(event: MouseEvent) => {
          if (event.target === event.currentTarget) { 
            setSelectedFurniture(null, null);
          }
        }}
      >
        <ambientLight intensity={1.5} /> 
        <directionalLight position={[10, 10, 5]} intensity={1} castShadow />
        <directionalLight position={[-10, 10, -5]} intensity={0.5} />

        {/* Room Box / Floor */}
        <Box args={[20, 0.2, 20]} position={[0, -0.1, 0]} receiveShadow>
          <meshStandardMaterial color="white" />
        </Box>
        {/* Optional: Add walls if you want a full box */}
        {/* Back Wall */}
        {/* <Box args={[20, 10, 0.2]} position={[0, 5, -10]} receiveShadow castShadow>
          <meshStandardMaterial color="#f0f0f0" />
        </Box> */}
        {/* Left Wall */}
        {/* <Box args={[0.2, 10, 20]} position={[-10, 5, 0]} receiveShadow castShadow>
          <meshStandardMaterial color="#f0f0f0" />
        </Box> */}

        <PerspectiveCamera makeDefault position={[5, 5, 10]} fov={50} />
        <OrbitControls makeDefault ref={orbitControlsRef} target={[0, 1, 0]} />

        {/* showroomModelUrl is not used for now */}
        {/* {showroomModelUrl && (
          <Suspense fallback={null}>
          </Suspense>
        )} */}

        <Selection>
          <EffectComposer multisampling={8} autoClear={false}>
            <Outline 
              visibleEdgeColor={0xff0000} // Bright Red for easy visibility
              edgeStrength={5}            // Moderate strength
              width={50}                  // Moderate width
              // blendFunction={BlendFunction.ADD} // Start without blendFunction, or try SCREEN if red is dark
            />
          </EffectComposer>

          {furnitureInScene.map(f => (
            <Suspense fallback={null} key={f.id}>
              <Furniture3DModel furniture={f} />
            </Suspense>
          ))}
        </Selection>
        
        {selectedObject && selectedFurnitureData && (
          <TransformControls
            ref={transformControlsRef}
            object={selectedObject}
            mode={selectedFurnitureData.transformMode || "translate"}
            onMouseUp={handleTransformEnd} 
          />
        )}
      </Canvas>
    </div>
  );
} 