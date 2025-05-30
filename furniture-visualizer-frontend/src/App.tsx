import './App.css'
// import { ThreeSceneViewer } from './components/ThreeSceneViewer' // 旧的单个模型查看器
import { useAppContext } from './contexts/AppContext'
// import { FileUpload } from './components/FileUpload' // FileUpload 现在在 FurnitureControls 内部
import { ShowroomViewer } from './components/ShowroomViewer';
import { FurnitureControls } from './components/FurnitureControls';

function App() {
  const {
    // modelPath, // modelPath 现在主要由 FurnitureControls 内部逻辑处理
    message, 
    // backendBaseUrl, // backendBaseUrl 从context中获取，并由组件内部使用
  } = useAppContext()

  return (
    <div className="App">
      <header className="App-header">
        <h1>家具可视化工具 - 样板房模式</h1>
      </header>
      
      <div className="main-content-area">
        <div className="controls-panel-area">
          <FurnitureControls />
        </div>
        <div className="viewer-area">
          <ShowroomViewer />
        </div>
      </div>

      {message && (
        <div className="message-container card global-message">
          {/* 
            考虑 message 的显示逻辑，modelPath 不再是判断成功与否的唯一标准，
            因为成功上传后是加入可用列表，不一定立即显示在viewer中。
            可以考虑在 context 中增加一个 messageType: 'success' | 'error' | 'info'
          */}
          <p className={`message ${message.toLowerCase().includes('成功') ? 'success' : message.toLowerCase().includes('失败') || message.toLowerCase().includes('错误') ? 'error' : 'info'}`}>{message}</p>
        </div>
      )}

      {/* 
      之前单个模型的查看器逻辑:
      <div className="viewer-container">
        <ThreeSceneViewer modelPath={modelPath} backendBaseUrl={backendBaseUrl} />
      </div> 
      */}
    </div>
  )
}

export default App
