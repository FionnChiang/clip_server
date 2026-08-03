import client from './client';
import type { Project } from '../types';

export async function listProjects(): Promise<Project[]> {
  const res = await client.get('/projects');
  return res.data;
}

export async function createProject(data: { name: string; description?: string; model_path?: string }): Promise<Project> {
  const res = await client.post('/projects', data);
  return res.data;
}

export async function getProject(id: string): Promise<Project> {
  const res = await client.get(`/projects/${id}`);
  return res.data;
}

export async function updateProject(id: string, data: Partial<Project>): Promise<Project> {
  const res = await client.put(`/projects/${id}`, data);
  return res.data;
}

export async function deleteProject(id: string): Promise<void> {
  await client.delete(`/projects/${id}`);
}
