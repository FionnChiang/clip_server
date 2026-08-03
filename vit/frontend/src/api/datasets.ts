import client from './client';
import type { Category, PaginatedImages, DatasetImage } from '../types';

export async function listCategories(projectId: string): Promise<Category[]> {
  const res = await client.get(`/projects/${projectId}/categories`);
  return res.data;
}

export async function addCategory(projectId: string, name: string): Promise<void> {
  await client.post(`/projects/${projectId}/categories`, { name });
}

export async function removeCategory(projectId: string, name: string): Promise<{ deleted_images: number }> {
  const res = await client.delete(`/projects/${projectId}/categories/${encodeURIComponent(name)}`);
  return res.data;
}

export async function uploadImages(
  projectId: string,
  files: File[],
  category: string
): Promise<{ uploaded: number; images: DatasetImage[] }> {
  const formData = new FormData();
  files.forEach((f) => formData.append('files', f));
  formData.append('category', category);
  const res = await client.post(`/projects/${projectId}/upload`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 120000,
  });
  return res.data;
}

export async function listImages(
  projectId: string,
  params: { category?: string; split?: string; page?: number; page_size?: number }
): Promise<PaginatedImages> {
  const res = await client.get(`/projects/${projectId}/images`, { params });
  return res.data;
}

export async function deleteImage(projectId: string, imageId: string): Promise<void> {
  await client.delete(`/projects/${projectId}/images/${imageId}`);
}

export async function applySplit(
  projectId: string,
  data: { train_ratio: number; seed: number }
): Promise<{ total_images: number }> {
  const res = await client.post(`/projects/${projectId}/split`, data);
  return res.data;
}
