import client from './client';
import type { SystemSettings, MySQLConfig, S3Config } from '../types';

export async function getSettings(): Promise<SystemSettings> {
  const res = await client.get('/settings');
  return res.data;
}

export async function updateMySQL(config: MySQLConfig): Promise<void> {
  await client.put('/settings/mysql', config);
}

export async function testMySQL(config: MySQLConfig): Promise<{ ok: boolean; message: string }> {
  const res = await client.post('/settings/test-mysql', config);
  return res.data;
}

export async function updateS3(config: S3Config): Promise<void> {
  await client.put('/settings/s3', config);
}

export async function testS3(): Promise<{ ok: boolean; message: string }> {
  const res = await client.post('/settings/test-s3');
  return res.data;
}
