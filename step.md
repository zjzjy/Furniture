
**项目名称（暂定）：** `furniture-visualizer-frontend`
(您可以自行更改)

**前端整体规划与模块分解 (MVP)**

1.  **项目初始化与基本设置**
    *   使用 Vite 创建 React + TypeScript 项目。
    *   安装必要的依赖 (`three`, `@react-three/fiber`, `@react-three/drei`, `axios`)。
    *   基本项目结构清理和配置（如 `tsconfig.json` 检查，`vite.config.ts` 可能的微调）。
    *   设置基础的CSS或样式方案（例如，一个简单的全局 `App.css`）。

2.  **API服务模块 (`src/services/apiService.ts`)**
    *   **功能**: 封装与后端API的通信。
    *   **接口**:
        *   `uploadImageAndConvertTo3D(file: File, depthScale?: number, thickness?: number): Promise<{ model_url: string; message: string }>`
            *   输入: 图片文件对象，可选的 `depthScale` 和 `thickness`。
            *   处理: 使用 `axios` 发送 `multipart/form-data` POST 请求到后端 `/api/v1/convert_to_3d`。
            *   输出: Promise，解析为后端返回的JSON对象。
            *   错误处理: 捕获axios错误并可能抛出自定义错误或返回特定错误结构。
    *   **对齐点**: 此模块的请求结构（URL、方法、参数名如 `file`, `depth_scale`, `thickness`）必须与后端 `main_api.py` 中定义的FastAPI端点完全一致。响应结构 (`model_url`, `message`) 也需一致。

3.  **核心状态管理 (使用 React Context + Hooks 或 Zustand)**
    *   **功能**: 管理应用级别的共享状态。对于MVP，React Context + Hooks 可能足够。如果未来状态复杂，可以考虑Zustand。
    *   **状态**:
        *   `isLoadingModel: boolean` (指示是否正在上传图片和等待模型转换)
        *   `modelUrl: string | null` (存储从API获取的GLB模型相对路径，如 `/static_glb_files/...glb`)
        *   `apiError: string | null` (存储API调用相关的错误信息)
        *   `backendBaseUrl: string` (存储后端API的基础URL，例如 `http://36.139.230.243:8000`)
    *   **操作 (Actions/Reducers/Setters)**:
        *   `setModelLoading(loading: boolean)`
        *   `setModelPath(path: string | null)` (注意这里是相对路径)
        *   `setApiError(error: string | null)`
    *   **对齐点**: 确保状态更新与API调用流程同步。`modelUrl` 存储的是后端返回的相对路径，前端在使用时会和 `backendBaseUrl` 拼接。

4.  **UI组件模块 (`src/components/`)**

    a.  **`FileUpload.tsx`**
        *   **功能**: 提供用户界面以上传图片。处理图片选择，触发API调用。
        *   **内部状态**: 可能有选中的文件对象。
        *   **交互**:
            *   点击按钮选择文件，或拖拽文件到指定区域。
            *   显示加载指示器（基于全局状态 `isLoadingModel`）。
            *   显示API错误（基于全局状态 `apiError`）。
        *   **接口 (Props)**: 无特殊外部props，主要与全局状态和`apiService`交互。
        *   **对齐点**: 调用 `apiService.uploadImageAndConvertTo3D`，并根据结果更新全局状态。

    b.  **`ThreeSceneViewer.tsx` (核心3D视图组件)**
        *   **功能**: 承载3D场景，加载和显示GLB模型，提供交互控制。
        *   **依赖**: `@react-three/fiber`, `@react-three/drei`。
        *   **内部组件/逻辑**:
            *   `<Canvas>`: R3F的画布。
            *   **Scene Setup**:
                *   `<PerspectiveCamera />` (或Drei的 `<CameraControls />` 增强版)。
                *   `<ambientLight />`, `<directionalLight />`。
                *   `<color attach="background" args={['#f0f0f0']} />` (简单背景)。
                *   (可选MVP+) 一个简单的地面平面 `<mesh><planeGeometry args={[10, 10]} /><meshStandardMaterial color="gray" /></mesh>`。
            *   **Model Loader (`Model.tsx` - 可作为子组件)**:
                *   **Props**: `url: string` (完整的模型URL)。
                *   **逻辑**: 使用 `@react-three/drei` 的 `useGLTF` hook 加载模型。
                *   渲染: `<primitive object={gltf.scene} scale={[1, 1, 1]} />` (可以调整初始缩放、位置、旋转)。
                *   错误和加载状态处理 (Drei的 `<Loader />` 或自定义)。
            *   **Controls**:
                *   `<OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />`
                *   (可选MVP+) `<TransformControls object={selectedMesh}>`: 需要逻辑来选中模型。
        *   **接口 (Props)**: `modelPath: string | null` (从全局状态获取，是相对路径)。
        *   **对齐点**: 从`modelPath`和全局状态`backendBaseUrl`构建完整的模型URL传递给`useGLTF`。

    c.  **`App.tsx` (主应用组件)**
        *   **功能**: 组合 `FileUpload` 和 `ThreeSceneViewer`。提供整体页面布局。包裹状态提供者 (Context Provider)。
        *   **对齐点**: 将全局状态（如 `modelUrl`, `isLoadingModel`, `apiError`）从Context中取出并传递给子组件，或让子组件直接从Context消费。

5.  **构建与部署 (后续)**
    *   `npm run build` 生成静态文件。
    *   可以将这些静态文件部署到任何静态网站托管服务，或者在您的AutoDL实例上用Nginx等服务。
    *   确保前端能正确访问后端API的URL（CORS已在后端配置为 `*`，但生产环境应收紧）。

**实现步骤与迭代：**

**迭代1: 项目设置和API连接测试**

1.  **任务**:
    *   Vite项目初始化 (`furniture-visualizer-frontend` 与 `react-ts`模板)。
    *   安装依赖。
    *   创建 `src/services/apiService.ts`。
    *   实现 `uploadImageAndConvertTo3D` 函数，硬编码后端URL (`http://36.139.230.243:8000`)。
    *   在 `App.tsx` 中添加一个简单的文件输入框和一个按钮。
    *   点击按钮时，获取文件，调用 `uploadImageAndConvertTo3D`，并将API的响应（成功或失败）打印到控制台。
2.  **目标**: 验证前端能够成功调用后端API并接收响应。
3.  **接口对齐**: `apiService.ts` 与后端API的请求/响应。

**迭代2: 基本3D场景与模型加载**

1.  **任务**:
    *   创建 `src/components/ThreeSceneViewer.tsx`。
    *   在 `ThreeSceneViewer.tsx` 中设置一个基本的R3F `<Canvas>`，包含相机、光照、`OrbitControls`。
    *   创建 `src/components/Model.tsx` 子组件。
        *   `Model.tsx` 接收一个完整的GLB模型URL作为prop。
        *   使用 `useGLTF` 加载并显示模型。
        *   使用 `@react-three/drei` 的 `<Loader />` 来显示加载进度（或 `Suspense`）。
    *   在 `App.tsx` 中，当迭代1的API调用成功并获取到 `model_url`后，将其（拼接上后端基础URL）传递给 `ThreeSceneViewer.tsx` (进而传递给 `Model.tsx`)。
2.  **目标**: 能够上传图片，API返回模型URL，前端加载并在3D场景中显示这个模型。
3.  **接口对齐**: `Model.tsx` 的 `url` prop；`App.tsx` 如何从API响应构造此URL。

**迭代3: 状态管理与UI完善**

1.  **任务**:
    *   实现全局状态管理 (例如 React Context)。
        *   定义状态：`isLoadingModel`, `modelUrl` (相对路径), `apiError`, `backendBaseUrl`。
        *   创建Context Provider并包裹 `App.tsx`。
    *   修改 `FileUpload.tsx`:
        *   从Context消费和更新 `isLoadingModel`, `apiError`, `setModelPath`。
        *   设计一个更友好的UI（例如拖放区）。
    *   修改 `ThreeSceneViewer.tsx`:
        *   从Context消费 `modelUrl` 和 `backendBaseUrl` 来构建完整URL。
        *   只有当 `modelUrl` 有效时才尝试加载和渲染 `Model.tsx`。
    *   在 `App.tsx` 中展示 `apiError` 给用户。
2.  **目标**: 应用具有更流畅的用户体验，能清晰地反映加载状态和错误。
3.  **接口对齐**: 组件与全局状态的交互。

**迭代4 (可选MVP+): 截图、家具控制等**

*   截图功能（从Canvas导出图像）。
*   `TransformControls` 实现家具的移动、旋转、缩放。
*   更精细的场景（毛坯房模型）。

我们将从 **迭代1** 开始。

**行动：**

1.  请在您的本地开发机器上打开终端。
2.  导航到您希望创建前端项目的父目录。
3.  准备好后，告诉我，我将提供Vite项目的初始化命令。

