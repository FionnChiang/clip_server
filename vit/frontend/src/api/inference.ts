import client from './client';
import type { PredictionResult } from '../types';

export async function predict(
  projectId: string,
  file: File,
  modelId?: string
): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append('file', file);
  if (modelId) formData.append('model_id', modelId);
  const res = await client.post(`/projects/${projectId}/predict`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 30000,
  });
  return res.data;
}

export async function predictTopK(
  projectId: string,
  file: File,
  k: number = 3,
  modelId?: string
): Promise<{ results: PredictionResult[] }> {
  const formData = new FormData();
  formData.append('file', file);
  if (modelId) formData.append('model_id', modelId);
  const res = await client.post(`/projects/${projectId}/predict/top-k?k=${k}`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 30000,
  });
  return res.data;
}
