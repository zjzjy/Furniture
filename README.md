施工中...
# Furniture Visualizer Frontend

This project is a **React + TypeScript + Vite** frontend for visualizing furniture. It works with a backend API to provide "one-click 2D furniture image to 3D model conversion and interactive placement in a virtual showroom".

## Features

- Upload a furniture image and generate a 3D GLB model via backend API.
- Adjustable model thickness parameter.
- Drag, rotate, and scale generated furniture models in a virtual showroom scene.
- Manage multiple furniture items and clear the scene.
- 3D rendering and interaction powered by Three.js (via @react-three/fiber and @react-three/drei).

## Project Structure

```
furniture-visualizer-frontend/
├── public/                # Static assets
├── src/
│   ├── components/        # React components (upload, 3D scene, controls, etc.)
│   ├── contexts/          # Global state management (AppContext)
│   ├── services/          # API service wrappers
│   ├── App.tsx            # Main app entry
│   ├── main.tsx           # React root
│   └── ...                # Styles, type definitions, etc.
├── package.json
├── tsconfig*.json
└── README.md
```