import React from 'react';
import { useAppContext } from '../contexts/AppContext';
import { FileUpload } from './FileUpload'; // 我们现有的文件上传组件
import { type IFurniture } from '../contexts/AppContext'; // 导入家具类型
import { v4 as uuidv4 } from 'uuid'; // 用于生成ID

export function FurnitureControls() {
  const {
    availableFurniture,
    addFurnitureToScene,
    // furnitureInScene, // 可能用于显示场景中的家具列表以进行编辑
    // selectedFurnitureId, setSelectedFurnitureId, // 用于选中和操作家具
    // removeFurnitureFromScene,
    // updateFurnitureInScene,
    modelPath, // 最新上传模型的路径
    backendBaseUrl,
    addAvailableFurniture,
    setModelPath, // 清除 modelPath 避免重复添加
    clearScene, // <--- 获取 clearScene 函数
  } = useAppContext();

  // 当 FileUpload 组件成功生成模型 (modelPath 更新) 时，将其添加到 availableFurniture
  // 这个效应应该在 modelPath 确实是一个新的、成功的路径时触发
  React.useEffect(() => {
    if (modelPath) { // modelPath 是类似 /static_glb_files/uuid.glb 的相对路径
      const newFurniture: IFurniture = {
        id: uuidv4(),
        name: `新家具 ${availableFurniture.length + 1}`,
        modelUrl: `${backendBaseUrl}${modelPath}`, // 需要完整的URL
        position: [0, 0, 0], // 默认初始位置
        rotation: [0, 0, 0],
        scale: [1, 1, 1],
      };
      addAvailableFurniture(newFurniture);
      setModelPath(null); // 清除，防止重复添加或用于其他逻辑
      // TODO: 可能需要一个更明确的信号表明模型已处理并添加到可用列表
    }
  }, [modelPath, backendBaseUrl, addAvailableFurniture, setModelPath, availableFurniture.length]);

  const handlePlaceFurniture = (furniture: IFurniture) => {
    // 简单实现：直接添加到场景中央，未来可以实现更复杂的放置逻辑
    const furnitureForScene = { 
      ...furniture, 
      id: uuidv4(), // 每次放置到场景都给一个新的ID，允许同一物品多次放置
      position: [Math.random()*2-1, 0, Math.random()*2-1] as [number,number,number] // 随机放置示例
    };
    addFurnitureToScene(furnitureForScene);
  };

  return (
    <div className="furniture-controls card">
      <h4>家具控制面板</h4>
      
      {/* 1. 上传新家具的模块 (使用现有的 FileUpload) */}
      <div className="upload-section">
        <h5>上传新家具模型</h5>
        <FileUpload />
      </div>

      {/* 2. 可用家具列表 */}
      <div className="available-furniture-section">
        <h5>可用家具库</h5>
        {availableFurniture.length === 0 && <p>上传图片以生成家具。</p>}
        <ul style={{ listStyleType: 'none', padding: 0 }}>
          {availableFurniture.map(item => (
            <li key={item.id} style={{ border: '1px solid #eee', marginBottom: '5px', padding: '5px' }}>
              {item.name} ({item.id.substring(0,6)})
              {/* TODO: 显示家具预览图 item.previewUrl */}
              <button onClick={() => handlePlaceFurniture(item)} style={{ marginLeft: '10px' }}>
                放置到场景
              </button>
            </li>
          ))}
        </ul>
      </div>

      {/* 新增场景控制部分 */}
      <div className="scene-controls-section" style={{ marginTop: '1rem', paddingTop: '1rem', borderTop: '1px solid #eee' }}>
        <h5>场景操作</h5>
        <button onClick={clearScene} style={{ backgroundColor: '#ff6961', color: 'white' }}>
          清空场景中所有家具
        </button>
      </div>

      {/* 3. 场景中家具的控制 (未来实现) */}
      {/* 
      <div className="scene-furniture-section">
        <h5>场景中的家具</h5>
        <p>TODO: 显示场景中的家具列表，允许选中、编辑、删除。</p>
      </div>
      */}
    </div>
  );
} 