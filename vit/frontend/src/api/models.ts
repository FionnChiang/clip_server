import client from './client';
import type { ModelInfo } from '../types';

export async function listModels(projectId: string): Promise<ModelInfo[]> {
  const res = await client.get(`/projects/${projectId}/models`);
  return res.data;
}

export async function deleteModel(projectId: string, modelId: string): Promise<void> {
  await client.delete(`/projects/${projectId}/models/${modelId}`);
}
