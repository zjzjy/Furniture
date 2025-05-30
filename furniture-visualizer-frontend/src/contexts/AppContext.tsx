import React, { createContext, useState, useContext, type ReactNode } from 'react';
import * as THREE from 'three'; // 导入 THREE

// 定义单个家具的类型
export interface IFurniture { // 导出以便其他组件可以使用
  id: string;
  name: string; // 家具名称，可以来自文件名或用户输入
  modelUrl: string; // 模型的相对路径
  previewUrl?: string; // 可选的预览图URL
  position: [number, number, number];
  rotation: [number, number, number]; // 欧拉角 (radians)
  scale: [number, number, number];
  transformMode?: 'translate' | 'rotate' | 'scale'; // 新增：变换模式
}

// 1. 定义状态的接口
interface IAppState {
  isLoading: boolean;            // 通用加载状态 (例如上传时)
  modelPath: string | null;      // 最新上传成功后模型的相对路径 (未来可能用于添加到家具列表)
  message: string;               // API相关的消息 (成功或错误)
  thickness: number;             // 用户选择的模型厚度 (用于新家具生成)
  backendBaseUrl: string;        // 后端基础URL
  
  // --- 样板房功能状态 ---
  showroomModelUrl: string | null;       // 样板房GLB模型的URL (可以是相对或绝对路径)
  furnitureInScene: IFurniture[];      // 当前场景中所有家具的列表
  selectedFurnitureId: string | null;  // 当前选中的家具ID
  selectedObject: THREE.Object3D | null; // 新增：当前选中的3D对象引用
  availableFurniture: IFurniture[];    // 用户上传/拥有但尚未放置到场景中的家具列表（暂定）
}

// 2. 定义Context提供的值的接口 (状态 + 修改状态的方法)
interface IAppContextProps extends IAppState {
  setIsLoading: (loading: boolean) => void;
  setModelPath: (path: string | null) => void;
  setMessage: (message: string) => void;
  setThickness: (thickness: number) => void;
  
  // --- 样板房功能setter ---
  setShowroomModelUrl: (url: string | null) => void;
  addFurnitureToScene: (item: IFurniture) => void;
  updateFurnitureInScene: (updatedItem: IFurniture) => void;
  removeFurnitureFromScene: (id: string) => void;
  setSelectedFurnitureId: (id: string | null) => void;
  setSelectedFurniture: (id: string | null, object: THREE.Object3D | null) => void;
  addAvailableFurniture: (item: IFurniture) => void; // 将新生成的家具添加到可用列表
  clearScene: () => void; // 新增：清空场景函数
}

// 3. 创建Context，初始值为undefined，因为Provider会提供实际值
const AppContext = createContext<IAppContextProps | undefined>(undefined);

// 4. 创建Provider组件
interface AppProviderProps {
  children: ReactNode;
}

export const AppProvider: React.FC<AppProviderProps> = ({ children }) => {
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [modelPath, setModelPath] = useState<string | null>(null); // 仍然用于指示最新上传结果
  const [message, setMessage] = useState<string>('');
  const [thickness, setThickness] = useState<number>(0.05);
  const backendBaseUrl = 'http://localhost:8000'; 

  // --- 样板房状态初始化 ---
  const [showroomModelUrl, setShowroomModelUrl] = useState<string | null>(null); // 示例: '/models/showroom.glb'
  const [furnitureInScene, setFurnitureInScene] = useState<IFurniture[]>([]);
  const [selectedFurnitureId, setSelectedFurnitureId] = useState<string | null>(null);
  const [selectedObject, setSelectedObject] = useState<THREE.Object3D | null>(null); // 新增状态
  const [availableFurniture, setAvailableFurniture] = useState<IFurniture[]>([]);

  // --- 样板房操作 ---
  const addFurnitureToScene = (item: IFurniture) => setFurnitureInScene(prev => [...prev, item]);
  
  const updateFurnitureInScene = (updatedItem: IFurniture) => {
    setFurnitureInScene(prev => prev.map(item => item.id === updatedItem.id ? updatedItem : item));
  };

  const removeFurnitureFromScene = (id: string) => {
    setFurnitureInScene(prev => prev.filter(item => item.id !== id));
    if (selectedFurnitureId === id) {
      setSelectedFurniture(null, null); // 使用统一的函数清空选中项
    }
  };

  const addAvailableFurniture = (item: IFurniture) => {
    setAvailableFurniture(prev => [...prev, item]);
  };

  const setSelectedFurniture = (id: string | null, object: THREE.Object3D | null) => {
    setSelectedFurnitureId(id);
    setSelectedObject(object);
  };

  const clearScene = () => { // 新增实现
    setFurnitureInScene([]);
    setSelectedFurniture(null, null);
    setMessage("场景已清空"); // 可选：给用户一个反馈
  };

  const contextValue: IAppContextProps = {
    isLoading,
    modelPath,
    message,
    thickness,
    backendBaseUrl,
    showroomModelUrl,
    furnitureInScene,
    selectedFurnitureId,
    selectedObject,
    availableFurniture,
    setIsLoading,
    setModelPath,
    setMessage,
    setThickness,
    setShowroomModelUrl,
    addFurnitureToScene,
    updateFurnitureInScene,
    removeFurnitureFromScene,
    setSelectedFurnitureId,
    setSelectedFurniture,
    addAvailableFurniture,
    clearScene, // 新增
  };

  return <AppContext.Provider value={contextValue}>{children}</AppContext.Provider>;
};

// 5. 创建自定义Hook以方便消费Context
export const useAppContext = (): IAppContextProps => {
  const context = useContext(AppContext);
  if (context === undefined) {
    throw new Error('useAppContext必须在AppProvider内部使用');
  }
  return context;
}; 