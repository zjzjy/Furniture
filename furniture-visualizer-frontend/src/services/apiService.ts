import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000'; // 后端API的基础URL (修改为本地地址)

interface ConvertTo3DResponse {
  model_url: string;
  message: string;
}

/**
 * 上传图片并将其转换为3D模型。
 * @param file 图片文件对象
 * @param depthScale 可选的深度缩放因子
 * @param thickness 可选的厚度
 * @returns Promise，解析为包含模型URL和消息的对象
 */
export const uploadImageAndConvertTo3D = async (
  file: File,
  depthScale?: number,
  thickness?: number
): Promise<ConvertTo3DResponse> => {
  const formData = new FormData();
  formData.append('file', file);
  if (depthScale !== undefined) {
    formData.append('depth_scale', depthScale.toString());
  }
  if (thickness !== undefined) {
    formData.append('thickness', thickness.toString());
  }

  try {
    const response = await axios.post<{ model_url: string; message: string }>(
      `${API_BASE_URL}/api/v1/convert_to_3d`,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      // 处理 Axios 错误
      console.error('API Error:', error.response?.data || error.message);
      throw new Error(error.response?.data?.detail || error.message || 'An unknown API error occurred');
    } else {
      // 处理其他类型的错误
      console.error('Unexpected Error:', error);
      throw new Error('An unexpected error occurred');
    }
  }
}; 