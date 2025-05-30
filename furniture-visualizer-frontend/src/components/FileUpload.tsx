import { useState, type ChangeEvent } from 'react';
import { useAppContext } from '../contexts/AppContext';
import { uploadImageAndConvertTo3D } from '../services/apiService';

export function FileUpload() {
  const {
    isLoading,
    thickness,
    // modelPath, // FileUpload 组件不需要直接使用 modelPath，它由 App 消费
    // message,   // message 也在 App 中显示，但 FileUpload 会设置它
    setIsLoading,
    setModelPath,
    setMessage,
    setThickness
  } = useAppContext();

  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setSelectedFile(event.target.files[0]);
      setMessage(''); 
      setModelPath(null);
    }
  };

  const handleThicknessChange = (event: ChangeEvent<HTMLInputElement>) => {
    setThickness(parseFloat(event.target.value));
  };

  const handleSubmit = async () => {
    if (!selectedFile) {
      setMessage('请先选择一个文件。');
      return;
    }
    setIsLoading(true);
    setModelPath(null);
    setMessage('正在上传和转换文件...');
    try {
      const response = await uploadImageAndConvertTo3D(selectedFile, undefined, thickness);
      console.log('API Response:', response);
      // Message 和 ModelPath 的设置由 Context 处理，App.tsx 会显示它们
      setMessage(`成功！模型URL: ${response.model_url}, 消息: ${response.message}`);
      setModelPath(response.model_url);
    } catch (error) {
      console.error('API Error:', error);
      setMessage(`转换失败: ${error instanceof Error ? error.message : '未知错误'}`);
      setModelPath(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="controls-container card"> {/* 类名与 App.tsx 中一致 */}
      <div>
        <label htmlFor="file-upload">选择图片:</label>
        <input 
          id="file-upload" 
          type="file" 
          accept="image/*" 
          onChange={handleFileChange} 
          disabled={isLoading} 
        />
      </div>
      
      <div className="thickness-control">
        <label htmlFor="thickness-slider">模型厚度: {thickness.toFixed(3)}</label>
        <input 
          id="thickness-slider" 
          type="range" 
          min="0.005" 
          max="0.5" 
          step="0.005" 
          value={thickness} 
          onChange={handleThicknessChange} 
          disabled={isLoading} 
        />
      </div>
      
      <button onClick={handleSubmit} disabled={!selectedFile || isLoading}>
        {isLoading ? '处理中...' : '上传并转换'}
      </button>
      {/* 消息显示将保留在App.tsx中，因为它更像是一个全局通知 */}
    </div>
  );
} 