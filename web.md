rembg onnxruntime-gpu time fastapi python-multipart
pip install 'uvicorn[standard]'
# MVP: 2D 图片转 3D 家具展示器

这是一个使用 HTML, CSS, 和 JavaScript (Three.js) 构建的最小可行产品前端。
它允许用户拖拽一张家具图片，发送到后端进行3D转换，然后在场景中展示返回的3D模型。

## HTML (`index.html`)

```html
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>2D to 3D MVP</title>
    <style>
        body { margin: 0; font-family: Arial, sans-serif; display: flex; flex-direction: column; align-items: center; background-color: #f0f0f0; }
        #header { padding: 10px; background-color: #333; color: white; text-align: center; width: 100%; }
        #drop-zone {
            width: 80%;
            max-width: 600px;
            height: 150px;
            border: 3px dashed #ccc;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 20px auto;
            text-align: center;
            font-size: 1.2em;
            color: #777;
            background-color: #fff;
        }
        #drop-zone.dragover {
            border-color: #333;
            background-color: #eee;
        }
        #scene-container {
            width: 80%;
            max-width: 800px;
            height: 500px; /* 初始高度，可以调整 */
            margin: 20px auto;
            border: 1px solid #ccc;
            background-color: #fff;
            position: relative; /* 用于加载指示器定位 */
        }
        #loading-indicator {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 1.5em;
            color: #555;
            display: none; /* 默认隐藏 */
            background-color: rgba(255, 255, 255, 0.8);
            padding: 20px;
            border-radius: 8px;
        }
        #error-message {
            color: red;
            margin-top: 10px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div id="header">
        <h1>家具图片转3D模型展示 (MVP)</h1>
    </div>

    <div id="drop-zone">
        <p>将家具图片拖拽到这里</p>
    </div>

    <div id="scene-container">
        <div id="loading-indicator">正在转换模型...</div>
    </div>
    <div id="error-message"></div>

    <!-- 引入 Three.js -->
    <script src="https://unpkg.com/three@0.139.2/build/three.min.js"></script>
    <!-- 引入 GLTFLoader -->
    <script src="https://unpkg.com/three@0.139.2/examples/js/loaders/GLTFLoader.js"></script>
    <!-- 引入 OrbitControls -->
    <script src="https://unpkg.com/three@0.139.2/examples/js/controls/OrbitControls.js"></script>

    <script>
        // JavaScript 代码将放在这里
    </script>
</body>
</html>
```

## CSS

CSS 已内联在 HTML 的 `<style>` 标签中以便于演示。主要包括：
- 页面基本布局和样式。
- 图片拖拽区域 (`#drop-zone`) 的样式及其拖拽悬浮时的样式 (`.dragover`)。
- 3D场景容器 (`#scene-container`) 的样式。
- 加载指示器 (`#loading-indicator`) 的样式，默认隐藏。
- 错误信息区域 (`#error-message`) 的样式。

## JavaScript (`<script>` 标签内)

JavaScript 代码将负责：
1.  初始化 Three.js 场景。
2.  处理图片拖拽事件。
3.  将图片发送到后端API。
4.  加载并显示后端返回的GLB模型。
5.  实现基本的场景交互。

--- 
下一步将填充 JavaScript 逻辑。 