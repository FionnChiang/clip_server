import client from './client';
import type { TrainingJob, TrainingMetric, TrainingConfigPayload } from '../types';

export async function listJobs(projectId: string): Promise<TrainingJob[]> {
  const res = await client.get(`/projects/${projectId}/jobs`);
  return res.data;
}

export async function startTraining(projectId: string, config: TrainingConfigPayload): Promise<TrainingJob> {
  const res = await client.post(`/projects/${projectId}/train`, config, { timeout: 60000 });
  return res.data;
}

export async function getJob(projectId: string, jobId: string): Promise<TrainingJob> {
  const res = await client.get(`/projects/${projectId}/jobs/${jobId}`);
  return res.data;
}

export async function stopTraining(projectId: string, jobId: string): Promise<void> {
  await client.post(`/projects/${projectId}/jobs/${jobId}/stop`);
}

export async function getJobMetrics(projectId: string, jobId: string): Promise<TrainingMetric[]> {
  const res = await client.get(`/projects/${projectId}/jobs/${jobId}/metrics`);
  return res.data;
}
