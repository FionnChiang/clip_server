import { Routes, Route, Navigate } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import ProjectLayoutPage from './pages/ProjectLayout';
import DatasetPage from './pages/DatasetPage';
import TrainingPage from './pages/TrainingPage';
import ModelsPage from './pages/ModelsPage';
import InferencePage from './pages/InferencePage';
import ServicesPage from './pages/ServicesPage';
import SettingsPage from './pages/SettingsPage';

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Dashboard />} />
      <Route path="/settings" element={<SettingsPage />} />
      <Route path="/projects/:projectId" element={<ProjectLayoutPage />}>
        <Route index element={<Navigate to="dataset" replace />} />
        <Route path="dataset" element={<DatasetPage />} />
        <Route path="training" element={<TrainingPage />} />
        <Route path="models" element={<ModelsPage />} />
        <Route path="inference" element={<InferencePage />} />
        <Route path="services" element={<ServicesPage />} />
      </Route>
    </Routes>
  );
}
